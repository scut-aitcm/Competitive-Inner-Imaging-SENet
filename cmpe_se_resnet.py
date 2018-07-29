# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import init
from mxnet import ndarray as nd
import math

net_se_ratio = 16
cmpe_se_ratio = 16

CMPESEBlockV2_kernel = (1, 1)

try:
    ctx = mx.gpu()
    _ = nd.array([0], ctx=ctx)
except:
    ctx = mx.cpu()


class PreActBottleneckSEBlock(HybridBlock):
    '''
        SE
    '''
    
    def __init__(self, channels, stride=1, downsample=False, **kwargs):
        super(PreActBottleneckSEBlock, self).__init__(**kwargs)
        self.expansion = 4
        self.downsample = downsample
        self.bn1 = nn.BatchNorm()
        self.conv1 = nn.Conv2D(channels=channels, kernel_size=1, use_bias=False,
                               weight_initializer=init.Normal(math.sqrt(2. / (1. * channels))))
        self.bn2 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels=channels, kernel_size=3, strides=stride, padding=1, use_bias=False,
                               weight_initializer=init.Normal(math.sqrt(2. / (9. * channels))))
        self.bn3 = nn.BatchNorm()
        self.conv3 = nn.Conv2D(channels=self.expansion * channels, kernel_size=1, use_bias=False,
                               weight_initializer=init.Normal(math.sqrt(2. / (1. * self.expansion * channels))))
        if downsample:
            self.shortcut = nn.HybridSequential()
            self.shortcut.add(
                nn.Conv2D(channels=self.expansion * channels, kernel_size=1, strides=stride, use_bias=False,
                          weight_initializer=init.Normal(math.sqrt(2. / (1. * self.expansion * channels))))
            )
        
        self.net_se_conv = nn.HybridSequential()
        self.net_se_conv.add(
            nn.GlobalAvgPool2D(),
            nn.Dense(self.expansion * channels / net_se_ratio, use_bias=False),
            nn.Activation(activation='relu'),
            nn.Dense(self.expansion * channels, activation='sigmoid', use_bias=False)
        )
    
    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.downsample else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        
        se_out = self.net_se_conv(out)
        se_out = se_out.reshape(shape=se_out.shape + (1, 1,))
        out = out * se_out + shortcut
        return out


class PreActBottleneckCMPESEBlockV1(HybridBlock):
    '''
        Double FC
    '''
    
    def __init__(self, channels, stride=1, downsample=False, **kwargs):
        super(PreActBottleneckCMPESEBlockV1, self).__init__(**kwargs)
        self.expansion = 4
        self.downsample = downsample
        self.bn1 = nn.BatchNorm()
        self.conv1 = nn.Conv2D(channels=channels, kernel_size=1, use_bias=False,
                               weight_initializer=init.Normal(math.sqrt(2. / (1. * channels))))
        self.bn2 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels=channels, kernel_size=3, strides=stride, padding=1, use_bias=False,
                               weight_initializer=init.Normal(math.sqrt(2. / (9. * channels))))
        self.bn3 = nn.BatchNorm()
        self.conv3 = nn.Conv2D(channels=self.expansion * channels, kernel_size=1, use_bias=False,
                               weight_initializer=init.Normal(math.sqrt(2. / (1. * self.expansion * channels))))
        if downsample:
            self.shortcut = nn.HybridSequential()
            self.shortcut.add(
                nn.Conv2D(channels=self.expansion * channels, kernel_size=1, strides=stride, use_bias=False,
                          weight_initializer=init.Normal(math.sqrt(2. / (1. * self.expansion * channels))))
            )
        
        self.net_se_input_conv = nn.HybridSequential()
        self.net_se_input_conv.add(
            nn.GlobalAvgPool2D(),
            nn.Conv2D(channels=self.expansion * channels / net_se_ratio, kernel_size=1, use_bias=False),
            nn.Activation('relu')
        )
        
        self.net_se_input_x = nn.HybridSequential()
        self.net_se_input_x.add(
            nn.GlobalAvgPool2D(),
            nn.Conv2D(channels=self.expansion * channels / net_se_ratio, kernel_size=1, use_bias=False),
            nn.Activation('relu')
        )
        
        self.net_se = nn.HybridSequential()
        self.net_se.add(
            nn.Conv2D(channels=self.expansion * channels, kernel_size=1, use_bias=False),
            nn.Activation(activation='sigmoid')
        )
    
    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.downsample else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        
        se_input_conv = self.net_se_input_conv(out)
        se_input_skipx = self.net_se_input_x(shortcut)
        se_input = nd.concat(se_input_conv, se_input_skipx, dim=1)
        se_out = self.net_se(se_input)
        out = out * se_out + shortcut
        return out


class PreActBottleneckCMPESEBlockV2(HybridBlock):
    '''
        Conv_2x1 or conv_1x1 pair-view
    '''
    
    def __init__(self, channels, stride=1, downsample=False, **kwargs):
        super(PreActBottleneckCMPESEBlockV2, self).__init__(**kwargs)
        self.expansion = 4
        self.downsample = downsample
        self.bn1 = nn.BatchNorm()
        self.conv1 = nn.Conv2D(channels=channels, kernel_size=1, use_bias=False,
                               weight_initializer=init.Normal(math.sqrt(2. / (1. * channels))))
        self.bn2 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels=channels, kernel_size=3, strides=stride, padding=1, use_bias=False,
                               weight_initializer=init.Normal(math.sqrt(2. / (9. * channels))))
        self.bn3 = nn.BatchNorm()
        self.conv3 = nn.Conv2D(channels=self.expansion * channels, kernel_size=1, use_bias=False,
                               weight_initializer=init.Normal(math.sqrt(2. / (1. * self.expansion * channels))))
        if downsample:
            self.shortcut = nn.HybridSequential()
            self.shortcut.add(
                nn.Conv2D(channels=self.expansion * channels, kernel_size=1, strides=stride, use_bias=False,
                          weight_initializer=init.Normal(math.sqrt(2. / (1. * self.expansion * channels))))
            )
        
        self.net_Global_skipx = nn.HybridSequential()
        self.net_Global_skipx.add(nn.GlobalAvgPool2D())
        self.net_Global_conv = nn.HybridSequential()
        self.net_Global_conv.add(nn.GlobalAvgPool2D())
        
        self.Multi_Map = nn.HybridSequential()
        self.Multi_Map.add(
            nn.Conv2D(channels=channels / cmpe_se_ratio, kernel_size=CMPESEBlockV2_kernel, use_bias=False),
            nn.BatchNorm()
        )
        
        self.net_SE = nn.HybridSequential()
        self.net_SE.add(
            nn.Flatten(),
            nn.Dense(self.expansion * channels / net_se_ratio, activation='relu', use_bias=False),
            nn.Dense(self.expansion * channels, activation='sigmoid', use_bias=False)
        )
    
    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.downsample else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        
        se_input_conv = self.net_Global_conv(out)
        se_input_conv = se_input_conv.reshape(shape=(se_input_conv.shape[0], 1, se_input_conv.shape[1], 1))
        se_input_skipx = self.net_Global_skipx(shortcut)
        se_input_skipx = se_input_skipx.reshape(shape=(se_input_skipx.shape[0], 1, se_input_skipx.shape[1], 1))
        
        conv_x_concat = nd.concat(se_input_conv, se_input_skipx, dim=-1)
        multi_map = self.Multi_Map(conv_x_concat)
        one_map = nd.mean(multi_map, axis=1, keepdims=True)
        se_out = self.net_SE(one_map)
        se_out = se_out.reshape(shape=se_out.shape + (1, 1,))
        
        out = out * se_out + shortcut
        return out


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


CMPE_SE_block_versions = [PreActBottleneckCMPESEBlockV1, PreActBottleneckCMPESEBlockV2]


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
    net = CMPESEResNet(PreActBottleneckSEBlock, layers, channels, **kwargs)
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
    global CMPESEBlockV2_kernel
    CMPESEBlockV2_kernel = (1, 1) if use_1x1 else (1, 2)
    return get_cmpe_se_resnet(version=2, num_layers=164, **kwargs)


# net = cmpe_se_v2_resnet164(use_1x1=True, classes=100)
