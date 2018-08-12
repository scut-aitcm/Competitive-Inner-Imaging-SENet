# -*- coding: utf-8 -*-

from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import init
import math
from src.block import cmpese_resnetv2_block


class CMPESEResNet(HybridBlock):
    def __init__(self, block, layers, channels, classes, **kwargs):
        super(CMPESEResNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.BatchNorm(scale=False, center=False))
            self.features.add(nn.Conv2D(channels=16, kernel_size=3, strides=1, padding=1, use_bias=False,
                                        weight_initializer=init.Normal(math.sqrt(2. / (9. * 16)))))
            
            # stage1
            self.features.add(
                self._make_layer(block, layers[0], channels[0], stride=1, stage_index=1, in_channels=16 / 4))
            
            # stage2
            self.features.add(
                self._make_layer(block, layers[1], channels[1], stride=2, stage_index=2, in_channels=channels[0]))
            
            # stage3
            self.features.add(
                self._make_layer(block, layers[2], channels[2], stride=2, stage_index=3, in_channels=channels[1]))
            
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.AvgPool2D(8))
            self.features.add(nn.Flatten())
            self.output = nn.Dense(classes)
    
    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, prefix=''))
            for _ in range(layers - 1):
                layer.add(block(channels, 1, False, prefix=''))
        return layer
    
    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


CMPE_SE_block_versions = [cmpese_resnetv2_block.PreActBottleneckCMPESEBlockV1,
                          cmpese_resnetv2_block.PreActBottleneckCMPESEBlockV2]


def _get_resnet_spec(num_layers):
    '''
    resnet164:  [18, 18, 18] , [16, 32, 64]
    '''
    assert (num_layers - 2) % 9 == 0
    n = (num_layers - 2) // 9
    channels = [16, 32, 64]
    layers = [n] * len(channels)
    return layers, channels


def get_se_resnet(num_layers, **kwargs):
    layers, channels = _get_resnet_spec(num_layers)
    net = CMPESEResNet(cmpese_resnetv2_block.PreActBottleneckSEBlock, layers, channels, **kwargs)
    return net


def get_cmpe_se_resnet(version, num_layers, **kwargs):
    layers, channels = _get_resnet_spec(num_layers)
    block_class = CMPE_SE_block_versions[version - 1]
    net = CMPESEResNet(block_class, layers, channels, **kwargs)
    return net


def cmpe_se_resnet164(**kwargs):
    return get_se_resnet(num_layers=164, **kwargs)


def cmpe_se_v1_resnet164(**kwargs):
    return get_cmpe_se_resnet(version=1, num_layers=164, **kwargs)


def cmpe_se_v2_resnet164(use_1x1=True, **kwargs):
    cmpese_resnetv2_block.CMPESEBlockV2_kernel = (1, 1) if use_1x1 else (1, 2)
    return get_cmpe_se_resnet(version=2, num_layers=164, **kwargs)

# net = cmpe_se_v2_resnet164(use_1x1=True, classes=10)
