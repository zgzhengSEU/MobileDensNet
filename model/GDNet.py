import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from torch import Tensor
from typing import Optional, Callable, List
from mmcv.cnn import ConvModule, is_norm
from mmcv.ops import ModulatedDeformConv2dPack as DCNv2
from mmcv.cnn import build_norm_layer, build_conv_layer
from mmengine.model import constant_init, normal_init


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible(
            (reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class GhostModuleV2(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, mode=None, args=None):
        super(GhostModuleV2, self).__init__()
        self.mode = mode
        self.gate_fn = nn.Sigmoid()

        if self.mode in ['original']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels*(ratio-1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size,
                          stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1,
                          dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        elif self.mode in ['attn']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels*(ratio-1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size,
                          stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1,
                          dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.short_conv = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride,
                          kernel_size//2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1,
                          padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1,
                          padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :]
        elif self.mode in ['attn']:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :]*F.interpolate(self.gate_fn(res), size=(out.shape[-2], out.shape[-1]), mode='nearest')


class GhostBottleneckV2(nn.Module):

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0., layer_id=None, args=None):
        super(GhostBottleneckV2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        if layer_id <= 1:
            self.ghost1 = GhostModuleV2(
                in_chs, mid_chs, relu=True, mode='original', args=args)
        else:
            self.ghost1 = GhostModuleV2(
                in_chs, mid_chs, relu=True, mode='attn', args=args)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size-1)//2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        self.ghost2 = GhostModuleV2(
            mid_chs, out_chs, relu=False, mode='original', args=args)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicSepConv(nn.Module):

    def __init__(self, in_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicSepConv, self).__init__()
        self.out_channels = in_planes
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=in_planes, bias=bias)
        self.bn = nn.BatchNorm2d(
            in_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(RFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        self.inter_planes = in_planes // 4

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, self.inter_planes, kernel_size=1, stride=1),
            BasicSepConv(self.inter_planes, kernel_size=3,
                         stride=1, padding=1, dilation=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, self.inter_planes, kernel_size=1, stride=1),
            BasicConv(self.inter_planes, self.inter_planes,
                      kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BasicSepConv(self.inter_planes, kernel_size=3,
                         stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, self.inter_planes, kernel_size=1, stride=1),
            BasicConv(self.inter_planes, self.inter_planes,
                      kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicSepConv(self.inter_planes, kernel_size=3,
                         stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, self.inter_planes //
                      2, kernel_size=1, stride=1),
            BasicConv(self.inter_planes//2, (self.inter_planes//4)*3,
                      kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv((self.inter_planes//4)*3, self.inter_planes,
                      kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicSepConv(self.inter_planes, kernel_size=3,
                         stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(
            4*self.inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        out = out*self.scale + x
        out = self.relu(out)

        return out


class OCRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, gamma=False):
        super(OCRFB, self).__init__()
        self.scale = nn.Parameter(torch.ones(1) * scale) if gamma else scale
        self.out_channels = out_planes
        self.inter_planes = in_planes // 4

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, self.inter_planes, kernel_size=1, stride=1),
            BasicSepConv(self.inter_planes, kernel_size=3,
                         stride=1, padding=1, dilation=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, self.inter_planes, kernel_size=1, stride=1),
            BasicConv(self.inter_planes, self.inter_planes,
                      kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BasicSepConv(self.inter_planes, kernel_size=3,
                         stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, self.inter_planes, kernel_size=1, stride=1),
            BasicConv(self.inter_planes, self.inter_planes,
                      kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicSepConv(self.inter_planes, kernel_size=3,
                         stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, self.inter_planes //
                      2, kernel_size=1, stride=1),
            BasicConv(self.inter_planes//2, (self.inter_planes//4)*3,
                      kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv((self.inter_planes//4)*3, self.inter_planes,
                      kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicSepConv(self.inter_planes, kernel_size=3,
                         stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch4 = nn.Sequential(
            BasicConv(in_planes, self.inter_planes, kernel_size=1, stride=1),
            BasicConv(self.inter_planes, self.inter_planes,
                      kernel_size=(5, 1), stride=1, padding=(2, 0)),
            BasicSepConv(self.inter_planes, kernel_size=3,
                         stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch5 = nn.Sequential(
            BasicConv(in_planes, self.inter_planes, kernel_size=1, stride=1),
            BasicConv(self.inter_planes, self.inter_planes,
                      kernel_size=(1, 5), stride=stride, padding=(0, 2)),
            BasicSepConv(self.inter_planes, kernel_size=3,
                         stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch6 = nn.Sequential(
            BasicConv(in_planes, self.inter_planes //
                      2, kernel_size=1, stride=1),
            BasicConv(self.inter_planes//2, (self.inter_planes//4)*3,
                      kernel_size=(1, 5), stride=1, padding=(0, 2)),
            BasicConv((self.inter_planes//4)*3, self.inter_planes,
                      kernel_size=(5, 1), stride=stride, padding=(2, 0)),
            BasicSepConv(self.inter_planes, kernel_size=3,
                         stride=1, padding=7, dilation=7, relu=False)
        )
        self.ConvLinear = BasicConv(
            7*self.inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch3(x)
        x5 = self.branch3(x)
        x6 = self.branch6(x)
        
        out = torch.cat((x0, x1, x2, x3, x4, x5, x6), 1)
        out = self.ConvLinear(out)
        out = out*self.scale + x
        out = self.relu(out)

        return out

class FEM(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 in_as_mid=False,
                 kernel_size=1,
                 use_dilation=False):
        super().__init__()
        # same padding
        if use_dilation:
            dilation = 2
        else:
            dilation = 1

        if kernel_size == 3:
            pad = 2
        else:
            pad = 0

        self.conv_stem = nn.Conv2d(in_channels, mid_channels if in_as_mid else in_channels,
                                   kernel_size=kernel_size, padding=pad, dilation=dilation)

        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv1 = nn.Conv2d(
            mid_channels if in_as_mid else in_channels, mid_channels, kernel_size=kernel_size, padding=pad,  dilation=dilation)
        self.maxpool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=kernel_size, padding=pad,  dilation=dilation)
        self.maxpool3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=kernel_size, padding=pad,  dilation=dilation)

        self.final_conv = nn.Conv2d(
            mid_channels * 3 + (mid_channels if in_as_mid else in_channels), out_channels, kernel_size=kernel_size, padding=pad,  dilation=dilation)

    def forward(self, x):
        shortcut = self.conv_stem(x)

        x = self.maxpool1(shortcut)
        x = self.conv1(x)
        shortcut1 = x
        x = self.maxpool2(x)
        x = self.conv2(x)
        shortcut2 = x
        x = self.maxpool3(x)
        x = self.conv3(x)
        shortcut3 = x

        out = torch.cat([shortcut, shortcut1, shortcut2, shortcut3], dim=1)
        out = self.final_conv(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for REB

    The Bottleneck contains three ConvLayers and one residual connection.

    Args:
        in_channels (int): The number of input channels.
        mid_channels (int): The number of middle output channels.
        dilation (int): Dilation rate.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 dilation,
                 use_dcn_mode=0,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.use_dcn_mode = use_dcn_mode

        self.conv1 = ConvModule(
            in_channels, mid_channels, 1, norm_cfg=norm_cfg)

        if use_dcn_mode == 0:
            self.conv2 = ConvModule(
                mid_channels,
                mid_channels,
                3,
                padding=dilation,
                dilation=dilation,
                norm_cfg=norm_cfg)
        elif use_dcn_mode == 1:
            self.conv2 = nn.Sequential(
                DCNv2(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False),
                build_norm_layer(norm_cfg, mid_channels)[1]
            )

        elif use_dcn_mode == 2:
            self.conv2 = nn.Sequential(
                ConvModule(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    norm_cfg=norm_cfg),
                DCNv2(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False),
                build_norm_layer(norm_cfg, mid_channels)[1]
            )
        elif use_dcn_mode == 3:
            self.conv2 = nn.Sequential(
                DCNv2(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False),
                build_norm_layer(norm_cfg, mid_channels)[1],
                ConvModule(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    norm_cfg=norm_cfg),
                DCNv2(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False),
                build_norm_layer(norm_cfg, mid_channels)[1]
            )
        elif use_dcn_mode == 4:
            self.conv2 = nn.Sequential(
                ConvModule(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    norm_cfg=norm_cfg),
                DCNv2(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False),
                build_norm_layer(norm_cfg, mid_channels)[1],
                DCNv2(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False),
                build_norm_layer(norm_cfg, mid_channels)[1]
            )
        elif use_dcn_mode == 5:
            self.conv2 = nn.Sequential(
                ConvModule(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    norm_cfg=norm_cfg),
                DCNv2(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    bias=False),
                build_norm_layer(norm_cfg, mid_channels)[1],
                DCNv2(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    bias=False),
                build_norm_layer(norm_cfg, mid_channels)[1]
            )
        self.conv3 = ConvModule(
            mid_channels, in_channels, 1, norm_cfg=norm_cfg)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out


class REB(nn.Module):
    """
    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
              which are 1x1 conv + 3x3 conv
        - the dilated residual block

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        block_mid_channels (int): The number of middle block output channels
        num_residual_blocks (int): The number of residual blocks.
        block_dilations (list): The list of residual blocks dilation.
    """

    def __init__(self, in_channels, out_channels, block_mid_channels,
                 num_residual_blocks, block_dilations, use_dcn_mode=0):
        super(REB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_mid_channels = block_mid_channels
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations = block_dilations
        self.use_dcn_mode = use_dcn_mode
        self._init_layers()

    def _init_layers(self):
        encoder_blocks = []
        for i in range(self.num_residual_blocks):
            dilation = self.block_dilations[i]
            encoder_blocks.append(
                Bottleneck(
                    self.out_channels,
                    self.block_mid_channels,
                    dilation=dilation,
                    use_dcn_mode=self.use_dcn_mode))
        self.REB_blocks = nn.Sequential(*encoder_blocks)

    def init_weights(self):
        for m in self.REB_blocks.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)

    def forward(self, x):
        return self.REB_blocks(x)


class ContextualModule(nn.Module):
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6)):
        super(ContextualModule, self).__init__()
        self.scales = []
        self.scales = nn.ModuleList(
            [self._make_scale(in_channels, size) for size in sizes])
        self.bottleneck = nn.Conv2d(
            in_channels * 2, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.weight_net = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self._initialize_weights()

    def __make_weight(self, feature, scale_feature):
        weight_feature = feature - scale_feature
        return F.sigmoid(self.weight_net(weight_feature))

    def _make_scale(self, in_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        multi_scales = [F.interpolate(input=stage(feats), size=(
            h, w), mode='bilinear', align_corners=True) for stage in self.scales]
        with torch.cuda.amp.autocast(enabled=False):
            weights = [self.__make_weight(feats.to(dtype=torch.float32), scale_feature.to(dtype=torch.float32)).to(dtype=feats.dtype)
                       for scale_feature in multi_scales]
        overall_features = [(multi_scales[0] * weights[0] + multi_scales[1]*weights[1] + multi_scales[2] *
                             weights[2] + multi_scales[3] * weights[3]) / (weights[0] + weights[1] + weights[2] + weights[3])] + [feats]
        bottle = self.bottleneck(torch.cat(overall_features, 1))
        return self.relu(bottle)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class GhostNetV2P3_RFB_CAN_REB(nn.Module):
    def __init__(self, FEM_kernel_size=1, use_dilation_in_FEM=False, use_CAN=True, use_se=True, use_dcn_mode=0, gamma=False, width=1.0, block=GhostBottleneckV2, args=None):
        super(GhostNetV2P3_RFB_CAN_REB, self).__init__()

        self.cfgs = [
            # k, t, c, SE, s
            # ====== p1 ==============
            # stage 0
            [[3,  16,  16, 0, 1]],  # 0
            # ====== p2 ==============
            # stage 1
            [[3,  48,  24, 0, 2]],
            # stage 2
            [[3,  72,  24, 0, 1]],  # 2
            # ====== p3 ==============
            # stage 3
            [[5,  72,  40, 0.25 if use_se else 0, 2]],
            # stage 4
            [[5, 120,  40, 0.25 if use_se else 0, 1]],  # 4
            # ====== p4 ==============
            # stage 5
            [[3, 240,  80, 0, 1]],
            # stage 6
            [[3, 200,  80, 0, 1],
             [3, 184,  80, 0, 1],
             [3, 184,  80, 0, 1],
             [3, 480, 112, 0.25 if use_se else 0, 1],
             [3, 672, 112, 0.25 if use_se else 0, 1]],  # 6
            # ====== p5 ==============
            # stage 7
            [[5, 672, 160, 0.25 if use_se else 0, 2]],
            # stage 8
            [[5, 960, 160, 0, 1],
             [5, 960, 160, 0.25 if use_se else 0, 1],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 0.25 if use_se else 0, 1]]]  # 8

        self.use_dcn_mode = use_dcn_mode
        self.use_CAN = use_CAN
        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        #block = block
        layer_id = 0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                if block == GhostBottleneckV2:
                    layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                  se_ratio=se_ratio, layer_id=layer_id, args=args))
                input_channel = output_channel
                layer_id += 1
            stages.append(nn.Sequential(*layers))

        self.blocks = nn.Sequential(*stages)

        if self.use_CAN:
            self.ContextualModule = ContextualModule(in_channels=_make_divisible(
                112 * width, 4), out_channels=_make_divisible(112 * width, 4))

        self.P2_RFB = OCRFB(in_planes=_make_divisible(
            24 * width, 4), out_planes=_make_divisible(24 * width, 4), scale=1.0, gamma=gamma)
        self.P3_RFB = OCRFB(in_planes=_make_divisible(
            112 * width, 4), out_planes=_make_divisible(112 * width, 4), scale=1.0, gamma=gamma)
        self.P4_RFB = OCRFB(in_planes=_make_divisible(
            160 * width, 4), out_planes=_make_divisible(160 * width, 4), scale=1.0, gamma=gamma)
        self.P2_REB_out = REB(in_channels=_make_divisible(24 * width, 4), out_channels=_make_divisible(
            24 * width, 4), block_mid_channels=_make_divisible(16 * width, 4), num_residual_blocks=4, block_dilations=[2, 4, 6, 8], use_dcn_mode=use_dcn_mode)
        self.P3_REB_out = REB(_make_divisible(112 * width, 4) + _make_divisible(40 * width, 4), out_channels=_make_divisible(
            112 * width, 4) + _make_divisible(40 * width, 4), block_mid_channels=_make_divisible(40 * width, 4), num_residual_blocks=4, block_dilations=[2, 4, 6, 8], use_dcn_mode=use_dcn_mode)
        # building last layer
        self.P2_FEM = FEM(in_channels=_make_divisible(24 * width, 4) + _make_divisible(112 * width, 4) + _make_divisible(40 * width, 4), in_as_mid=True,
                          mid_channels=_make_divisible(24 * width, 4), out_channels=_make_divisible(24 * width, 4), kernel_size=FEM_kernel_size, use_dilation=use_dilation_in_FEM)
        self.P3_FEM = FEM(in_channels=_make_divisible(112 * width, 4) + _make_divisible(40 * width, 4), mid_channels=_make_divisible(40 * width, 4),
                          out_channels=_make_divisible(112 * width, 4) + _make_divisible(40 * width, 4), kernel_size=FEM_kernel_size, use_dilation=use_dilation_in_FEM)
        self.P4_FEM = FEM(in_channels=_make_divisible(160 * width, 4), mid_channels=_make_divisible(80 * width, 4),
                          out_channels=_make_divisible(40 * width, 4), kernel_size=FEM_kernel_size, use_dilation=use_dilation_in_FEM)
        self.output_layer_p3 = ConvModule(_make_divisible(
            112 * width, 4) + _make_divisible(40 * width, 4), 1, kernel_size=1, norm_cfg=dict(type='BN', requires_grad=True))
        self.output_layer_p2 = ConvModule(
            _make_divisible(24 * width, 4), 1, kernel_size=1, norm_cfg=dict(type='BN', requires_grad=True))

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        feat = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in [2, 6, 8]:
                feat.append(x)
        p2 = self.P2_RFB(feat[0])
        if self.use_CAN:
            p3 = self.ContextualModule(self.P3_RFB(feat[1]))
        else:
            p3 = self.P3_RFB(feat[1])
        p4 = self.P4_RFB(feat[2])

        p4 = self.P4_FEM(p4)
        p4_up = F.interpolate(
            p4, size=(p3.shape[2], p3.shape[3]), mode='bilinear', align_corners=True)

        p3 = self.P3_FEM(torch.cat([p3, p4_up], dim=1))
        p3_up = F.interpolate(
            p3, size=(p2.shape[2], p2.shape[3]), mode='bilinear', align_corners=True)

        p2 = self.P2_FEM(torch.cat([p2, p3_up], dim=1))

        p2_out = self.output_layer_p2(self.P2_REB_out(p2))
        p3_out = self.output_layer_p3(self.P3_REB_out(p3))

        return p3_out, p2_out


if __name__ == '__main__':
    model = GhostNetV2P3_RFB_CAN_REB(
        use_dilation_in_FEM=False, use_CAN=True, use_se=True, use_dcn_mode=4, width=1.6).to('cuda')
    # model = GhostNetV2P3_RFB(use_dilation=False, width=1.6).to('cuda')
    # checkpoint_path = '/home/gp.sc.cc.tohoku.ac.jp/duanct/openmmlab/GhostDensNet/checkpoints/ghostnetv2_torch/ck_ghostnetv2_16.pth.tar'
    # load_checkpoint(model, checkpoint_path, strict=False, map_location='cuda')
    model.eval()
    input = torch.ones((1, 3, 1920, 1080)).to('cuda')
    # p3_out, p2_out = model(input)
    # print(p3_out.shape)
    # print(p2_out.shape)
    # print(model)
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    from fvcore.nn import flop_count_str
    flops = FlopCountAnalysis(model, input)
    print(f'input shape: {input.shape}')
    print(flop_count_table(flops))
