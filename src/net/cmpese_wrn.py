# -*- coding: utf-8 -*-

from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from src.block import cmpese_wrn_block


class CIFARWideResNet(HybridBlock):
    def __init__(self, block, layers, channels, classes, **kwargs):
        super(CIFARWideResNet, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.BatchNorm(scale=False, center=False))
            
            self.features.add(
                nn.Conv2D(channels[0], 3, 1, 1, use_bias=False))
            self.features.add(nn.BatchNorm())
            
            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(
                    self._make_layer(block, num_layer, channels[i + 1], stride, i + 1, in_channels=in_channels))
                in_channels = channels[i + 1]
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.AvgPool2D(8))
            self.features.add(nn.Flatten())
            self.output = nn.Dense(classes)
    
    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels,
                            in_channels=in_channels, prefix=''))
            for _ in range(layers - 1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix=''))
        return layer
    
    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


CMPE_SE_block_versions = [cmpese_wrn_block.CMPESEBlockDoubleFC,
                          cmpese_wrn_block.CMPESEBlock1x1,
                          cmpese_wrn_block.CMPESEBlock2x1,
                          cmpese_wrn_block.CMPESEBlock3x3,
                          ]


def _get_wrn_spec(num_layers, width_factor):
    assert (num_layers - 4) % 6 == 0
    n = (num_layers - 4) // 6
    layers = [n] * 3
    channels = [16, 16 * width_factor, 32 * width_factor, 64 * width_factor]
    return layers, channels


def get_se_wrn(num_layers, width_factor, **kwargs):
    layers, channels = _get_wrn_spec(num_layers, width_factor)
    net = CIFARWideResNet(cmpese_wrn_block.SEBlock, layers, channels, **kwargs)
    return net


def get_cmpe_se_wrn(version, num_layers, width_factor, **kwargs):
    layers, channels = _get_wrn_spec(num_layers, width_factor)
    block_class = CMPE_SE_block_versions[version]
    net = CIFARWideResNet(block_class, layers, channels, **kwargs)
    return net


def se_wrn28_10(**kwargs):
    return get_se_wrn(num_layers=28, width_factor=10, **kwargs)


def cmpe_se_doublefc_wrn28_10(**kwargs):
    return get_cmpe_se_wrn(version=0, num_layers=28, width_factor=10, **kwargs)


def cmpe_se_1x1_wrn28_10(**kwargs):
    return get_cmpe_se_wrn(version=1, num_layers=28, width_factor=10, **kwargs)


def cmpe_se_2x1_wrn28_10(**kwargs):
    return get_cmpe_se_wrn(version=2, num_layers=28, width_factor=10, **kwargs)


def cmpe_se_3x3_wrn28_10(**kwargs):
    return get_cmpe_se_wrn(version=3, num_layers=28, width_factor=10, **kwargs)

# net = cmpe_se_3x3_wrn28_10(classes=10)
