import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from torch import Tensor
from typing import Optional, Callable, List
from mmcv.cnn import ConvModule, is_norm
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


class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_a, self).__init__()
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
    """Bottleneck block for DilatedEncoder used in `YOLOF.

    <https://arxiv.org/abs/2103.09460>`.

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
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.conv1 = ConvModule(
            in_channels, mid_channels, 1, norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            mid_channels,
            mid_channels,
            3,
            padding=dilation,
            dilation=dilation,
            norm_cfg=norm_cfg)
        self.conv3 = ConvModule(
            mid_channels, in_channels, 1, norm_cfg=norm_cfg)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out


class DilatedEncoder(nn.Module):
    """Dilated Encoder for YOLOF <https://arxiv.org/abs/2103.09460>`.

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
                 num_residual_blocks, block_dilations):
        super(DilatedEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_mid_channels = block_mid_channels
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations = block_dilations
        self._init_layers()

    def _init_layers(self):
        encoder_blocks = []
        for i in range(self.num_residual_blocks):
            dilation = self.block_dilations[i]
            encoder_blocks.append(
                Bottleneck(
                    self.out_channels,
                    self.block_mid_channels,
                    dilation=dilation))
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def init_weights(self):
        for m in self.dilated_encoder_blocks.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)

    def forward(self, x):
        return self.dilated_encoder_blocks(x)


class GhostNetV2P3_RFB_DE(nn.Module):
    def __init__(self, FEM_kernel_size=1, use_dilation=False, width=1.0, block=GhostBottleneckV2, args=None):
        super(GhostNetV2P3_RFB_DE, self).__init__()
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
            [[5,  72,  40, 0.25, 2]],
            # stage 4
            [[5, 120,  40, 0.25, 1]],  # 4
            # ====== p4 ==============
            # stage 5
            [[3, 240,  80, 0, 1]],
            # stage 6
            [[3, 200,  80, 0, 1],
             [3, 184,  80, 0, 1],
             [3, 184,  80, 0, 1],
             [3, 480, 112, 0.25, 1],
             [3, 672, 112, 0.25, 1]],  # 6
            # ====== p5 ==============
            # stage 7
            [[5, 672, 160, 0.25, 2]],
            # stage 8
            [[5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1]]]  # 8

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

        self.P2_RFB = BasicRFB_a(in_planes=_make_divisible(
            24 * width, 4), out_planes=_make_divisible(24 * width, 4), scale=1.0)
        self.P3_RFB = BasicRFB_a(in_planes=_make_divisible(
            112 * width, 4), out_planes=_make_divisible(112 * width, 4), scale=1.0)
        self.P4_RFB = BasicRFB_a(in_planes=_make_divisible(
            160 * width, 4), out_planes=_make_divisible(160 * width, 4), scale=1.0)
        self.P2_DilatedEncoder_out = DilatedEncoder(in_channels=_make_divisible(24 * width, 4), out_channels=_make_divisible(
            24 * width, 4), block_mid_channels=_make_divisible(16 * width, 4), num_residual_blocks=4, block_dilations=[2, 4, 6, 8])
        self.P3_DilatedEncoder_out = DilatedEncoder(_make_divisible(112 * width, 4) + _make_divisible(40 * width, 4), out_channels=_make_divisible(
            112 * width, 4) + _make_divisible(40 * width, 4), block_mid_channels=_make_divisible(40 * width, 4), num_residual_blocks=4, block_dilations=[2, 4, 6, 8])
        # building last layer
        self.P2_FEM = FEM(in_channels=_make_divisible(24 * width, 4) + _make_divisible(112 * width, 4) + _make_divisible(40 * width, 4), in_as_mid=True,
                          mid_channels=_make_divisible(24 * width, 4), out_channels=_make_divisible(24 * width, 4), kernel_size=FEM_kernel_size, use_dilation=use_dilation)
        self.P3_FEM = FEM(in_channels=_make_divisible(112 * width, 4) + _make_divisible(40 * width, 4), mid_channels=_make_divisible(40 * width, 4),
                          out_channels=_make_divisible(112 * width, 4) + _make_divisible(40 * width, 4), kernel_size=FEM_kernel_size, use_dilation=use_dilation)
        self.P4_FEM = FEM(in_channels=_make_divisible(160 * width, 4), mid_channels=_make_divisible(80 * width, 4),
                          out_channels=_make_divisible(40 * width, 4), kernel_size=FEM_kernel_size, use_dilation=use_dilation)
        self.output_layer_p3 = nn.Conv2d(_make_divisible(
            112 * width, 4) + _make_divisible(40 * width, 4), 1, kernel_size=1)
        self.output_layer_p2 = nn.Conv2d(
            _make_divisible(24 * width, 4), 1, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[2:4]
        h //= 8
        w //= 8

        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        feat = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in [2, 6, 8]:
                feat.append(x)
        p2 = self.P2_RFB(feat[0])
        p3 = self.P3_RFB(feat[1])
        p4 = self.P4_RFB(feat[2])

        p4 = self.P4_FEM(p4)
        p4_up = F.interpolate(
            p4, size=(p3.shape[2], p3.shape[3]), mode='bilinear', align_corners=True)

        p3 = self.P3_FEM(torch.cat([p3, p4_up], dim=1))
        p3_up = F.interpolate(
            p3, size=(p2.shape[2], p2.shape[3]), mode='bilinear', align_corners=True)

        p2 = self.P2_FEM(torch.cat([p2, p3_up], dim=1))

        p2_out = self.output_layer_p2(self.P2_DilatedEncoder_out(p2))
        p3_out = self.output_layer_p3(self.P3_DilatedEncoder_out(p3))

        # out = F.interpolate(p2, size=(h, w), mode='bilinear', align_corners=True)
        return p3_out, p2_out


def _initialize_weights(model: nn.Module) -> nn.Module:
    """
    This function initialises the parameters of `model`.
    Supported layers:
    - Conv2d
    - ConvTranspose2d
    - Batchnorm2d
    - Linear
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    return model


class ConvNormActivation(nn.Sequential):
    """
    This snippet is adapted from `ConvNormActivation` provided by torchvision.
    Configurable block used for Convolution-Normalization-Activation blocks.
    Args:
        - `in_channels` (`int`): number of channels in the input image.
        - `out_channels` (`int`): number of channels produced by the Convolution-Normalization-Activation block.
        - `kernel_size`: (`int`, optional): size of the convolving kernel.
            - Default: `3`
        - `stride` (`int`, optional): stride of the convolution.
            - Default: `1`
        - `padding` (`int`, `tuple` or `str`, optional): padding added to all four sides of the input.
            - Default: `None`, in which case it will calculated as `padding = (kernel_size - 1) // 2 * dilation`.
        - `groups` (`int`, optional): number of blocked connections from input channels to output channels.
            - Default: `1`
        - `norm_layer` (`Callable[..., torch.nn.Module]`, optional): norm layer that will be stacked on top of the convolution layer. If `None` this layer won't be used.
            - Default: `torch.nn.BatchNorm2d`.
        - `activation_layer` (`Callable[..., torch.nn.Module]`, optional): activation function which will be stacked on top of the       normalization layer (if not `None`), otherwise on top of the `conv` layer. If `None` this layer wont be used.
            - Default: `torch.nn.ReLU6`
        - `dilation` (`int`): spacing between kernel elements.
            - Default: `1`
        - `inplace` (`bool`): parameter for the activation layer, which can optionally do the operation in-place.
            - Default `True`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[...,
                                            nn.Module]] = nn.ReLU6(inplace=True)
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=norm_layer is None,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer)
        super().__init__(*layers)
        self.out_channels = out_channels


class FeatureFuser(nn.Module):
    """
    This module fuses features with different receptive field sizes.
    1. Feat1 -> Feat1*
    2. Feat2 & Feat1* -> Weight2
       Feat3 & Feat1* -> Weight3
       ...
    3. Feat1* | (Feat2 * Weight2 + Feat3 * Weight3 + ...)
    4. Bottleneck.
    Args:
        - `in_channels_list` (`list[int]`): a list of the number of each feature's channels. `in_channels_list[0]` should be the number of channels of the feature from a pooling layer, while others are numbers of channels of features from conv layers. The number of output channel of this block is `in_channels_list[0]`
        - `batch_norm` (`bool`, optional): whether to use batch normalisation or not.
            - Default: `True`.
    """

    def __init__(self, in_channels_list: List[int], batch_norm: bool = True) -> None:
        super(FeatureFuser, self).__init__()
        if batch_norm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = None

        for idx, c in enumerate(in_channels_list):
            # Pooling layer.
            if idx == 0:
                num_1 = c
            # The first conv layer.
            elif idx == 1:
                num_2 = c
            # Other conv layers.
            else:
                assert num_2 == c

        # Increase the number of channels of Feat1.
        prior_conv = ConvNormActivation(
            in_channels=num_1,
            out_channels=num_2,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.ReLU(inplace=True)
        )
        self.prior_conv = _initialize_weights(prior_conv)

        # Conv layer for weight generation.
        weight_net = nn.Conv2d(
            in_channels=num_2,
            out_channels=num_2,
            kernel_size=1,
        )
        self.weight_net = _initialize_weights(weight_net)

        # Bottleneck layer.
        posterior_conv = ConvNormActivation(
            in_channels=num_2 * 2,
            out_channels=num_2,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.ReLU(inplace=True)
        )
        self.posterior_conv = _initialize_weights(posterior_conv)

    def __make_weights__(self, feat: Tensor, scaled_feat: Tensor) -> Tensor:
        return torch.sigmoid(self.weight_net(feat - scaled_feat))

    def forward(self, feats: List[Tensor]) -> Tensor:
        feat_0, feat_1 = feats[0], feats[1]

        # Increase the number of channels.
        feat_0 = self.prior_conv(feat_0)

        # Generate weights.
        weights = [self.__make_weights__(feat_1, feat_0)]

        # Fuse all features.
        feats = [sum([feat_0 * weights[0]]) / sum(weights)] + [feat_1]
        feats = torch.cat(feats, dim=1)

        # Reduce the number of channels.
        feats = self.posterior_conv(feats)

        return feats


class GhostNetV2P3_RFB_DE_Fusion(nn.Module):
    def __init__(self, batch_norm=False, FEM_kernel_size=1, use_dilation=False, width=1.0, block=GhostBottleneckV2, args=None):
        super(GhostNetV2P3_RFB_DE_Fusion, self).__init__()
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
            [[5,  72,  40, 0.25, 2]],
            # stage 4
            [[5, 120,  40, 0.25, 1]],  # 4
            # ====== p4 ==============
            # stage 5
            [[3, 240,  80, 0, 1]],
            # stage 6
            [[3, 200,  80, 0, 1],
             [3, 184,  80, 0, 1],
             [3, 184,  80, 0, 1],
             [3, 480, 112, 0.25, 1],
             [3, 672, 112, 0.25, 1]],  # 6
            # ====== p5 ==============
            # stage 7
            [[5, 672, 160, 0.25, 2]],
            # stage 8
            [[5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1]]]  # 8

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

        self.fuser_p2 = FeatureFuser([_make_divisible(
            24 * width, 4), _make_divisible(24 * width, 4)], batch_norm=batch_norm)
        self.fuser_p3 = FeatureFuser([_make_divisible(
            40 * width, 4), _make_divisible(112 * width, 4)], batch_norm=batch_norm)
        self.fuser_p4 = FeatureFuser([_make_divisible(
            160 * width, 4), _make_divisible(160 * width, 4)], batch_norm=batch_norm)

        self.P2_RFB = BasicRFB_a(in_planes=_make_divisible(
            24 * width, 4), out_planes=_make_divisible(24 * width, 4), scale=1.0)
        self.P3_RFB = BasicRFB_a(in_planes=_make_divisible(
            112 * width, 4), out_planes=_make_divisible(112 * width, 4), scale=1.0)
        self.P4_RFB = BasicRFB_a(in_planes=_make_divisible(
            160 * width, 4), out_planes=_make_divisible(160 * width, 4), scale=1.0)
        self.P2_DilatedEncoder_out = DilatedEncoder(in_channels=_make_divisible(24 * width, 4), out_channels=_make_divisible(
            24 * width, 4), block_mid_channels=_make_divisible(16 * width, 4), num_residual_blocks=4, block_dilations=[2, 4, 6, 8])
        self.P3_DilatedEncoder_out = DilatedEncoder(_make_divisible(112 * width, 4) + _make_divisible(40 * width, 4), out_channels=_make_divisible(
            112 * width, 4) + _make_divisible(40 * width, 4), block_mid_channels=_make_divisible(40 * width, 4), num_residual_blocks=4, block_dilations=[2, 4, 6, 8])
        # building last layer
        self.P2_FEM = FEM(in_channels=_make_divisible(24 * width, 4) + _make_divisible(112 * width, 4) + _make_divisible(40 * width, 4), in_as_mid=True,
                          mid_channels=_make_divisible(24 * width, 4), out_channels=_make_divisible(24 * width, 4), kernel_size=FEM_kernel_size, use_dilation=use_dilation)
        self.P3_FEM = FEM(in_channels=_make_divisible(112 * width, 4) + _make_divisible(40 * width, 4), mid_channels=_make_divisible(40 * width, 4),
                          out_channels=_make_divisible(112 * width, 4) + _make_divisible(40 * width, 4), kernel_size=FEM_kernel_size, use_dilation=use_dilation)
        self.P4_FEM = FEM(in_channels=_make_divisible(160 * width, 4), mid_channels=_make_divisible(80 * width, 4),
                          out_channels=_make_divisible(40 * width, 4), kernel_size=FEM_kernel_size, use_dilation=use_dilation)
        self.output_layer_p3 = nn.Conv2d(_make_divisible(
            112 * width, 4) + _make_divisible(40 * width, 4), 1, kernel_size=1)
        self.output_layer_p2 = nn.Conv2d(
            _make_divisible(24 * width, 4), 1, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[2:4]
        h //= 8
        w //= 8

        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        feats = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            feats.append(x)

        p2, p3, p4 = [feats[1], feats[2]], [
            feats[3], feats[6]], [feats[7], feats[8]]

        p2 = self.fuser_p2(p2)
        p3 = self.fuser_p3(p3)
        p4 = self.fuser_p4(p4)

        p2 = self.P2_RFB(p2)
        p3 = self.P3_RFB(p3)
        p4 = self.P4_RFB(p4)

        p4 = self.P4_FEM(p4)
        p4_up = F.interpolate(
            p4, size=(p3.shape[2], p3.shape[3]), mode='bilinear', align_corners=True)

        p3 = self.P3_FEM(torch.cat([p3, p4_up], dim=1))
        p3_up = F.interpolate(
            p3, size=(p2.shape[2], p2.shape[3]), mode='bilinear', align_corners=True)

        p2 = self.P2_FEM(torch.cat([p2, p3_up], dim=1))

        p2_out = self.output_layer_p2(self.P2_DilatedEncoder_out(p2))
        p3_out = self.output_layer_p3(self.P3_DilatedEncoder_out(p3))

        # out = F.interpolate(p2, size=(h, w), mode='bilinear', align_corners=True)
        return p3_out, p2_out


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
        weights = [self.__make_weight(feats, scale_feature)
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


class GhostNetV2P3_RFB_DE_CAN(nn.Module):
    def __init__(self, FEM_kernel_size=1, use_dilation=False, width=1.0, block=GhostBottleneckV2, args=None):
        super(GhostNetV2P3_RFB_DE_CAN, self).__init__()
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
            [[5,  72,  40, 0.25, 2]],
            # stage 4
            [[5, 120,  40, 0.25, 1]],  # 4
            # ====== p4 ==============
            # stage 5
            [[3, 240,  80, 0, 1]],
            # stage 6
            [[3, 200,  80, 0, 1],
             [3, 184,  80, 0, 1],
             [3, 184,  80, 0, 1],
             [3, 480, 112, 0.25, 1],
             [3, 672, 112, 0.25, 1]],  # 6
            # ====== p5 ==============
            # stage 7
            [[5, 672, 160, 0.25, 2]],
            # stage 8
            [[5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1]]]  # 8

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

        self.ContextualModule = ContextualModule(in_channels=_make_divisible(112 * width, 4), out_channels=_make_divisible(112 * width, 4))
        
        self.P2_RFB = BasicRFB_a(in_planes=_make_divisible(
            24 * width, 4), out_planes=_make_divisible(24 * width, 4), scale=1.0)
        self.P3_RFB = BasicRFB_a(in_planes=_make_divisible(
            112 * width, 4), out_planes=_make_divisible(112 * width, 4), scale=1.0)
        self.P4_RFB = BasicRFB_a(in_planes=_make_divisible(
            160 * width, 4), out_planes=_make_divisible(160 * width, 4), scale=1.0)
        self.P2_DilatedEncoder_out = DilatedEncoder(in_channels=_make_divisible(24 * width, 4), out_channels=_make_divisible(
            24 * width, 4), block_mid_channels=_make_divisible(16 * width, 4), num_residual_blocks=4, block_dilations=[2, 4, 6, 8])
        self.P3_DilatedEncoder_out = DilatedEncoder(_make_divisible(112 * width, 4) + _make_divisible(40 * width, 4), out_channels=_make_divisible(
            112 * width, 4) + _make_divisible(40 * width, 4), block_mid_channels=_make_divisible(40 * width, 4), num_residual_blocks=4, block_dilations=[2, 4, 6, 8])
        # building last layer
        self.P2_FEM = FEM(in_channels=_make_divisible(24 * width, 4) + _make_divisible(112 * width, 4) + _make_divisible(40 * width, 4), in_as_mid=True,
                          mid_channels=_make_divisible(24 * width, 4), out_channels=_make_divisible(24 * width, 4), kernel_size=FEM_kernel_size, use_dilation=use_dilation)
        self.P3_FEM = FEM(in_channels=_make_divisible(112 * width, 4) + _make_divisible(40 * width, 4), mid_channels=_make_divisible(40 * width, 4),
                          out_channels=_make_divisible(112 * width, 4) + _make_divisible(40 * width, 4), kernel_size=FEM_kernel_size, use_dilation=use_dilation)
        self.P4_FEM = FEM(in_channels=_make_divisible(160 * width, 4), mid_channels=_make_divisible(80 * width, 4),
                          out_channels=_make_divisible(40 * width, 4), kernel_size=FEM_kernel_size, use_dilation=use_dilation)
        self.output_layer_p3 = nn.Conv2d(_make_divisible(
            112 * width, 4) + _make_divisible(40 * width, 4), 1, kernel_size=1)
        self.output_layer_p2 = nn.Conv2d(
            _make_divisible(24 * width, 4), 1, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[2:4]
        h //= 8
        w //= 8

        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        feat = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in [2, 6, 8]:
                feat.append(x)
        p2 = self.P2_RFB(feat[0])
        p3 = self.ContextualModule(self.P3_RFB(feat[1]))
        p4 = self.P4_RFB(feat[2])

        p4 = self.P4_FEM(p4)
        p4_up = F.interpolate(
            p4, size=(p3.shape[2], p3.shape[3]), mode='bilinear', align_corners=True)

        p3 = self.P3_FEM(torch.cat([p3, p4_up], dim=1))
        p3_up = F.interpolate(
            p3, size=(p2.shape[2], p2.shape[3]), mode='bilinear', align_corners=True)

        p2 = self.P2_FEM(torch.cat([p2, p3_up], dim=1))

        p2_out = self.output_layer_p2(self.P2_DilatedEncoder_out(p2))
        p3_out = self.output_layer_p3(self.P3_DilatedEncoder_out(p3))

        # out = F.interpolate(p2, size=(h, w), mode='bilinear', align_corners=True)
        return p3_out, p2_out


if __name__ == '__main__':
    model = GhostNetV2P3_RFB_DE_CAN(
        use_dilation=False, width=1.6).to('cuda')
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
