"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""
import torch.nn.functional as F
import torch.nn as nn
import torch
import math


__all__ = ['mobilenetv3_large', 'mobilenetv3_small']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)


class BottomUPBlock_Cat(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
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

        self.conv = nn.Conv2d(in_channels, mid_channels,
                              kernel_size=kernel_size, padding=pad, dilation=dilation)

        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=kernel_size, padding=pad,  dilation=dilation)
        self.maxpool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=kernel_size, padding=pad,  dilation=dilation)
        self.maxpool3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=kernel_size, padding=pad,  dilation=dilation)

        self.final_conv = nn.Conv2d(
            mid_channels * 4, out_channels, kernel_size=kernel_size, padding=pad,  dilation=dilation)
        
    def forward(self, x):
        shortcut = self.conv(x)
        x = self.maxpool1(x)
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
    
    
class MobileNetV3Dens(nn.Module):
    def __init__(self, mode, kernel_size=1, use_dilation=False, width_mult=1.):
        super(MobileNetV3Dens, self).__init__()
        # setting of inverted residual blocks
        assert mode in ['large', 'small']
        if mode == 'large':
            self.cfgs = [
                # k, t, c, SE, HS, s 
                [3,   1,  16, 0, 0, 1], # p1 1
                [3,   4,  24, 0, 0, 2], 
                [3,   3,  24, 0, 0, 1], # p2 3
                [5,   3,  40, 1, 0, 2], 
                [5,   3,  40, 1, 0, 1],
                [5,   3,  40, 1, 0, 1], # p3 6
                [3,   6,  80, 0, 1, 2],
                [3, 2.5,  80, 0, 1, 1],
                [3, 2.3,  80, 0, 1, 1],
                [3, 2.3,  80, 0, 1, 1],
                [3,   6, 112, 1, 1, 1],
                [3,   6, 112, 1, 1, 1], # p4 12
                [5,   6, 160, 1, 1, 2],
                [5,   6, 160, 1, 1, 1],
                [5,   6, 160, 1, 1, 1]  # p5 15
            ]
            self.feat_layer = [6, 12, 15]
        else:
            self.cfgs = [
                # k, t, c, SE, HS, s 
                [3,    1,  16, 1, 0, 2], # p2 1
                [3,  4.5,  24, 0, 0, 2], 
                [3, 3.67,  24, 0, 0, 1], # p3 3
                [5,    4,  40, 1, 1, 2],
                [5,    6,  40, 1, 1, 1],
                [5,    6,  40, 1, 1, 1],
                [5,    3,  48, 1, 1, 1],
                [5,    3,  48, 1, 1, 1], # p4 8
                [5,    6,  96, 1, 1, 2],
                [5,    6,  96, 1, 1, 1],
                [5,    6,  96, 1, 1, 1], # p5 11
            ]
            self.feat_layer = [3, 8, 11]
        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        
        # bottomUP
        self.bottomUP_p5_to_p4 = BottomUPBlock_Cat(
            in_channels=160, mid_channels=160, out_channels=112, kernel_size=kernel_size, use_dilation=use_dilation)
        self.bottomUP_p4_to_p3 = BottomUPBlock_Cat(
            in_channels=112 * 2, mid_channels=112, out_channels=40, kernel_size=kernel_size, use_dilation=use_dilation)
        self.bottomUP_p3_to_out = BottomUPBlock_Cat(
            in_channels=40 * 2, mid_channels=40, out_channels=40, kernel_size=kernel_size, use_dilation=use_dilation)

        # building last layer
        self.output_layer = nn.Conv2d(40, 1, kernel_size=1)
        # building last several layers
        
        self._initialize_weights()

    def forward(self, x):
        h, w = x.shape[2:4]
        h //= 8
        w //= 8
        
        feat = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feat_layer:
                feat.append(x)
        
        p5 = self.bottomUP_p5_to_p4(feat[2])
        p5_up = F.interpolate(p5, size=(
            feat[1].shape[2], feat[1].shape[3]), mode='bilinear', align_corners=True)
        p4 = self.bottomUP_p4_to_p3(torch.cat([feat[1], p5_up], dim=1))
        p4_up = F.interpolate(p4, size=(
            feat[0].shape[2], feat[0].shape[3]), mode='bilinear', align_corners=True)
        p3 = self.bottomUP_p3_to_out(torch.cat([feat[0], p4_up], dim=1))
        
        out = self.output_layer(p3)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class MobileNetV3DensNew(nn.Module):
    def __init__(self, mode, kernel_size=1, use_dilation=False, width_mult=1.):
        super(MobileNetV3DensNew, self).__init__()
        # setting of inverted residual blocks
        assert mode in ['large', 'small']
        if mode == 'large':
            self.cfgs = [
                # k,  t,  c, SE, HS, s 
                [3,   1,  16, 0, 0, 1], # p1 1
                [3,   4,  24, 0, 0, 2], 
                [3,   3,  24, 0, 0, 1], # p2 3
                [5,   3,  40, 1, 0, 2], 
                [5,   3,  40, 1, 0, 1],
                [5,   3,  40, 1, 0, 1], # p3 6
                [3,   6,  80, 0, 1, 1],
                [3, 2.5,  80, 0, 1, 1],
                [3, 2.3,  80, 0, 1, 1],
                [3, 2.3,  80, 0, 1, 1],
                [3,   6, 112, 1, 1, 1],
                [3,   6, 112, 1, 1, 1], # p4 12
                [5,   6, 160, 1, 1, 1],
                [5,   6, 160, 1, 1, 1],
                [5,   6, 160, 1, 1, 1]  # p5 15
            ]
        else:
            self.cfgs = [
                # k, t, c, SE, HS, s 
                [3,    1,  16, 1, 0, 2], # p2 1
                [3,  4.5,  24, 0, 0, 2], 
                [3, 3.67,  24, 0, 0, 1], # p3 3
                [5,    4,  40, 1, 1, 1],
                [5,    6,  40, 1, 1, 1],
                [5,    6,  40, 1, 1, 1],
                [5,    3,  48, 1, 1, 1],
                [5,    3,  48, 1, 1, 1], # p4 8
                [5,    6,  96, 1, 1, 1],
                [5,    6,  96, 1, 1, 1],
                [5,    6,  96, 1, 1, 1], # p5 11
            ]
        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        
        # bottomUP
        # building last layer
        self.output_layer = nn.Conv2d(160, 1, kernel_size=1)
        # building last several layers
        
        self._initialize_weights()

    def forward(self, x):
        h, w = x.shape[2:4]
        h //= 8
        w //= 8
        
        x = self.features(x)
        out = self.output_layer(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_backend_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                            padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class MobileNetV3DensNew_dila(nn.Module):
    def __init__(self, mode, width_mult=1.):
        super(MobileNetV3DensNew_dila, self).__init__()
        # setting of inverted residual blocks
        assert mode in ['large', 'small']
        if mode == 'large':
            self.cfgs = [
                # k,  t,  c, SE, HS, s 
                [3,   1,  16, 0, 0, 1], # p1 1
                [3,   4,  24, 0, 0, 2], 
                [3,   3,  24, 0, 0, 1], # p2 3
                [5,   3,  40, 1, 0, 2], 
                [5,   3,  40, 1, 0, 1],
                [5,   3,  40, 1, 0, 1], # p3 6
                [3,   6,  80, 0, 1, 1],
                [3, 2.5,  80, 0, 1, 1],
                [3, 2.3,  80, 0, 1, 1],
                [3, 2.3,  80, 0, 1, 1],
                [3,   6, 112, 1, 1, 1],
                [3,   6, 112, 1, 1, 1], # p4 12
                [5,   6, 160, 1, 1, 1],
                [5,   6, 160, 1, 1, 1],
                [5,   6, 160, 1, 1, 1]  # p5 15
            ]
        else:
            self.cfgs = [
                # k, t, c, SE, HS, s 
                [3,    1,  16, 1, 0, 2], # p2 1
                [3,  4.5,  24, 0, 0, 2], 
                [3, 3.67,  24, 0, 0, 1], # p3 3
                [5,    4,  40, 1, 1, 1],
                [5,    6,  40, 1, 1, 1],
                [5,    6,  40, 1, 1, 1],
                [5,    3,  48, 1, 1, 1],
                [5,    3,  48, 1, 1, 1], # p4 8
                [5,    6,  96, 1, 1, 1],
                [5,    6,  96, 1, 1, 1],
                [5,    6,  96, 1, 1, 1], # p5 11
            ]
        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        
        self.backend_feat = [160, 160, 160, 120, 80, 40]
        self.backend = make_backend_layers(
            self.backend_feat, in_channels=160, dilation=True)
        # building last layer
        self.output_layer = nn.Conv2d(40, 1, kernel_size=1)
        # building last several layers
        
        self._initialize_weights()

    def forward(self, x):
        h, w = x.shape[2:4]
        h //= 8
        w //= 8
        
        x = self.features(x)
        x = self.backend(x)
        out = self.output_layer(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == '__main__':
    model = MobileNetV3DensNew_dila(mode='large').to('cuda')
    checkpoint_path = '/home/gp.sc.cc.tohoku.ac.jp/duanct/openmmlab/GhostDensNet/checkpoints/ghostnetv2_torch/ck_ghostnetv2_10.pth'
    model.eval()
    input_img = torch.ones((1, 3, 1920, 1080)).to('cuda')
    out = model(input_img)
    print(out.shape)
    # print(model)
    showstat = True
    if showstat:
        from torchstat import stat
        stat(model.to('cpu'), (3, 1920, 1080))
