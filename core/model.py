"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from core.resnet import Resnet18


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='reflect',
                 sn=False, n_a=False, only_act=False):
        super(GatedConv2d, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
            self.mask_conv2d = SpectralNorm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
        self.sigmoid = torch.nn.Sigmoid()
        self.activation = nn.ELU()
        self.norm = nn.InstanceNorm2d(out_channels)
        self.n_a = n_a
        self.only_act = only_act

    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask
        if self.n_a and self.only_act:
            x = self.activation(x)
        else:
            x = self.norm(x)
            x = self.activation(x)
        return x


class TransposeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
                 sn=False, scale_factor=2):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, sn,
                                        n_a=True)

    def forward(self, x, s=None):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.gated_conv2d(x)
        return x


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, pad_type='zero', actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self.pad_type = pad_type
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        if self.downsample:
            self.conv1 = GatedConv2d(dim_in, dim_in, 3, 2, 1, pad_type=self.pad_type)
        else:
            self.conv1 = GatedConv2d(dim_in, dim_in, 3, 1, 1, pad_type=self.pad_type)
        self.conv2 = GatedConv2d(dim_in, dim_out, 3, 1, 1, pad_type=self.pad_type)
        self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
        self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size =ks,
                stride =stride,
                padding =padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up  # x8, x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64,
                 actv=nn.LeakyReLU(0.2), upsample=False, pre_train=False):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self.pre_train = pre_train
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = GatedConv2d(dim_in, dim_out, 3, 1, 1, pad_type='zero')
        self.conv2 = GatedConv2d(dim_out, dim_out, 3, 1, 1, pad_type='zero')
        if self.pre_train:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_out, affine=True)
        else:
            self.norm1 = AdaIN(style_dim, dim_in)
            self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        if self.pre_train:
            x = self.norm1(x)
            x = self.actv(x)
            if self.upsample:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.conv1(x)
            x = self.norm2(x)
            x = self.actv(x)
            x = self.conv2(x)
        else:
            x = self.norm1(x, s)
            x = self.actv(x)
            if self.upsample:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.conv1(x)
            x = self.norm2(x, s)
            x = self.actv(x)
            x = self.conv2(x)
        return x

    def forward(self, x, s=None):
        out = self._residual(x, s)
        return out


class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=True):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class HighPass(nn.Module):
    def __init__(self, device):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]).to(device))

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-8, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)  # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class CoarseNet(nn.Module):
    def __init__(self, dim_in):
        super(CoarseNet, self).__init__()
        self.coarse = nn.Sequential(
            # encoder
            GatedConv2d(4, dim_in, 7, 1, 3, n_a=True, only_act=True),
            GatedConv2d(dim_in, dim_in * 2, 5, 2, 2, n_a=True),
            GatedConv2d(dim_in * 2, dim_in * 4, 3, 1, 1, n_a=True),
            GatedConv2d(dim_in * 4, dim_in * 4, 5, 2, 2, n_a=True),
            # Bottleneck
            GatedConv2d(dim_in * 4, dim_in * 4, 3, 1, 1, n_a=True),
            GatedConv2d(dim_in * 4, dim_in * 4, 3, 1, 1, n_a=True),
            GatedConv2d(dim_in * 4, dim_in * 4, 3, 1, 2, dilation=2, n_a=True),
            GatedConv2d(dim_in * 4, dim_in * 4, 3, 1, 4, dilation=4, n_a=True),
            GatedConv2d(dim_in * 4, dim_in * 4, 3, 1, 8, dilation=8, n_a=True),
            GatedConv2d(dim_in * 4, dim_in * 4, 3, 1, 16, dilation=16, n_a=True),
            GatedConv2d(dim_in * 4, dim_in * 4, 3, 1, 1, n_a=True),
            GatedConv2d(dim_in * 4, dim_in * 4, 3, 1, 1, n_a=True),

            # decoder
            TransposeGatedConv2d(dim_in * 4, dim_in * 2, 3, 1, 1),
            GatedConv2d(dim_in * 2, dim_in * 2, 3, 1, 1, n_a=True),
            TransposeGatedConv2d(dim_in * 2, dim_in, 3, 1, 1),
        )
        self.out_put = nn.Sequential(
            GatedConv2d(dim_in, dim_in, 3, 1, 1, sn=False),
            AttentionRefinementModule(dim_in, dim_in),
            nn.Conv2d(dim_in, 19, 1, 1, 0),
        )

    def forward(self, x, mask):
        x = torch.cat((x, mask), dim=1)
        x = self.coarse(x)
        x = self.out_put(x)
        return x


class Generator(nn.Module):
    def __init__(self, args, img_size=256, style_dim=64, max_conv_dim=512, pre=False):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.coarse = CoarseNet(dim_in)
        self.from_rgb = GatedConv2d(23, dim_in, 3, 1, 1, pad_type='zero') # 23
        self.pre = pre
        self.encode = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))
        self.out_put = nn.Sequential(
            GatedConv2d(dim_in, dim_in, 3, 1, 1),
            AttentionRefinementModule(dim_in, dim_in),
            nn.Conv2d(dim_in, 19, 1, 1, 0),
        )
        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        for iii in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            if iii == 0:
                self.decode.insert(
                    0, AdainResBlk(dim_out, dim_in, style_dim, upsample=True, pre_train=True))
            else:
                self.decode.insert(
                    0, AdainResBlk(dim_out, dim_in, style_dim, upsample=True, pre_train=pre))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        """
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf, pre_train=pre))
        """
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, pre_train=pre))
        # bottleneck fine blocks
        # lay_dilation = 1
        # for _ in range(4):
        #     lay_dilation *= 2
        # self.bottleneck.append(GatedConv2d(dim_out, dim_out, 3, 1, lay_dilation, dilation=lay_dilation, pad_type=args.pad_type, sn=True, n_a=True))

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_out, nhead=4)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer_layer = transformer_encoder
        device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        # self.hpf = HighPass(device)

    def forward(self, masked_img, s, masks):
        with torch.no_grad():
            first_out = self.coarse(masked_img, masks)
            first_out_soft = F.gumbel_softmax(first_out, dim=1)
        # first_out_soft = F.softmax(first_out, dim=1)
        # first_out_soft = first_out.argmax(1)
        # first_out_soft = torch.unsqueeze(first_out_soft, 1)
        second_in = torch.cat([masked_img, first_out_soft, masks], dim=1)
        x = self.from_rgb(second_in)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)
        b, c, h, w = x.shape
        feature0 = x.flatten(-2).permute(0, 2, 1)
        feature0 = self.transformer_layer(feature0)
        x = feature0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        for block in self.decode:
            if self.pre:
                x = block(x, None)
            else:
                x = block(x, s)
            if x.size(2) in [32, 64, 128]:
                mask = F.interpolate(masks, size=x.size(2), mode='bilinear')
                # x = x + self.hpf((1-mask) * cache[x.size(2)])
                x = x + (1 - mask) * cache[x.size(2)]

        return first_out, self.out_put(x), self.to_rgb(x)


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(4, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, mask, y):
        x = torch.cat((x, mask), dim=1)
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out


class PatchDiscriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super(PatchDiscriminator, self).__init__()
        # Down sampling
        dim_in = 2 ** 14 // img_size
        blocks = []
        blocks += [Conv2dLayer(4, dim_in, 7, 1, 3, norm='none')]
        for _ in range(6):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [Conv2dLayer(dim_in, dim_out, 4, 2, 1, norm='in')]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        # blocks += [nn.Sigmoid()]
        self.main = nn.Sequential(*blocks)

    def forward(self, img, mask, y):
        x = torch.cat((img, mask), 1)
        out = self.main(x)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]
        return out


def build_model(args, pre=False):
    # generator = nn.DataParallel(torch.compile(Generator(args, args.img_size, args.style_dim, pre=pre)))
    generator = nn.DataParallel(Generator(args, args.img_size, args.style_dim, pre=pre))
    style_encoder = nn.DataParallel(StyleEncoder(args.img_size, args.style_dim, args.num_domains))
    mapping_network = nn.DataParallel(MappingNetwork(args.latent_dim, args.latent_channels, args.num_domains))
    # discriminator = nn.DataParallel(Discriminator(args.img_size, args.num_domains))
    patchdiscriminator = nn.DataParallel(PatchDiscriminator(args.img_size, args.num_domains))
    # style_encoder = torch.compile(style_encoder, backend="inductor")
    # mapping_network = torch.compile(mapping_network, backend="inductor")
    nets = Munch(generator=generator,
                 style_encoder=style_encoder,
                 mapping_network=mapping_network,
                 # discriminator=discriminator,
                 discriminator=patchdiscriminator)

    return nets
