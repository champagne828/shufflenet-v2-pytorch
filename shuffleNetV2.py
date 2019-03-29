from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import math


class ShuffleNetV2(nn.Module):
    def __init__(self, symbol=1):
        super(ShuffleNetV2, self).__init__()
        self.module_channel = [24, 116, 232, 464, 1024]
        self.module_repeat = [4, 8, 4]
        if symbol == 2:
            self.module_channel = [64, 244, 488, 976, 1952, 2048]
            self.module_repeat = [3, 4, 6, 3]
        elif symbol == 3:
            self.module_channel = [64, 340, 680, 1360, 2720, 2048]
            self.module_repeat = [10, 10, 23, 10]
        self.features = nn.Sequential()
        ### Initial block
        self.features.add_module('init_conv', _ConvBnRelu(3, self.module_channel[0], 3, 2, 1))
        if symbol == 3:
            self.features.add_module('init_conv_s1', _ConvBnRelu(3, self.module_channel[0], 3, 1, 1))
            self.features.add_module('init_conv_s2', _ConvBnRelu(3, 2*self.module_channel[0], 3, 1, 1))
        self.features.add_module('init_pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ### Shuffle block
        for i in range(len(self.module_repeat)):
            self.add_block(i)
        ### Last block
        self.features.add_module('last_conv', _ConvBnRelu(self.module_channel[-2], self.module_channel[-1], 1, 1, 0))
        self.features.add_module('last_pool', nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(self.module_channel[-1], 1000)
        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def add_block(self, i):
        block = _ShuffleBlock(self.module_channel[i], self.module_channel[i+1], self.module_repeat[i])
        self.features.add_module('shuffle_block_%d' % (i + 1), block)
    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


class _ConvBnRelu(nn.Module):
    def __init__(self, inc, outc, k, s, p):
        super(_ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class _ShuffleBlock(nn.Sequential):
    def __init__(self, inc, outc, block_size):
        super(_ShuffleBlock, self).__init__()
        for i in range(block_size):
            if i == 0:
                layer = _DownsampleShuffleLayer(inc, outc//2)
            else:
                layer = _ShuffleLayer(outc//2)
            self.add_module('shuffle_layer_%d' % (i + 1), layer)


class _DownsampleShuffleLayer(nn.Module):
    def __init__(self, inc, outc):
        super(_DownsampleShuffleLayer, self).__init__()
        self.left_branch = nn.Sequential()
        self.left_branch.add_module('left_dwconv', _DWConvBn(inc, 2))
        self.left_branch.add_module('left_conv', _ConvBnRelu(inc, outc, 1, 1, 0))
        self.right_branch = nn.Sequential()
        self.right_branch.add_module('right_conv_head', _ConvBnRelu(inc, outc, 1, 1, 0))
        self.right_branch.add_module('right_dwconv', _DWConvBn(outc, 2))
        self.right_branch.add_module('right_conv_tail', _ConvBnRelu(outc, outc, 1, 1, 0))
    def forward(self, x):
        x1 = self.left_branch(x)
        x2 = self.right_branch(x)
        x = torch.cat((x1, x2), 1)
        return _channel_shuffle(x, 2)


class _ShuffleLayer(nn.Module):
    def __init__(self, channel):
        super(_ShuffleLayer, self).__init__()
        self.right_branch = nn.Sequential()
        self.right_branch.add_module('right_conv_head', _ConvBnRelu(channel, channel, 1, 1, 0))
        self.right_branch.add_module('right_dwconv', _DWConvBn(channel, 1))
        self.right_branch.add_module('right_conv_tail', _ConvBnRelu(channel, channel, 1, 1, 0))
    def forward(self, x):
        x1 = x[:, :(x.shape[1]//2), :, :]
        x2 = x[:, (x.shape[1]//2):, :, :]
        x = torch.cat((x1, self.right_branch(x2)), 1)
        return _channel_shuffle(x, 2)


class _DWConvBn(nn.Module):
    def __init__(self, channel, s):
        super(_DWConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=s, padding=1, groups=channel, bias=False)
        self.bn = nn.BatchNorm2d(channel)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


def _channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups   
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x
