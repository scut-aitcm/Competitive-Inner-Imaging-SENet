# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import ndarray as nd
from mxnet import init
from src import utils

net_se_ratio = 16
cmpe_se_ratio = 16

CMPESEBlockV2_kernel = (1, 1)

try:
    ctx = mx.gpu()
    _ = nd.array([0], ctx=ctx)
except:
    ctx = mx.cpu()


def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1, use_bias=False, in_channels=in_channels)


class SEBlock(HybridBlock):
    def __init__(self, channels, stride, downsample=False, drop_rate=0.0, in_channels=0, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = _conv3x3(channels, stride, in_channels)
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv3x3(channels, 1, channels)
        self.droprate = drop_rate
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False, in_channels=in_channels)
        else:
            self.downsample = None
        
        self.net_se_conv = nn.HybridSequential()  # se的输入的conv部分
        self.net_se_conv.add(
            nn.GlobalAvgPool2D(),
            nn.Dense(channels / net_se_ratio, use_bias=False),
            nn.Activation(activation='relu'),
            nn.Dense(channels, activation='sigmoid', use_bias=False)
        )
    
    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        conv_out = self.conv1(x)
        conv_out = self.bn2(conv_out)
        conv_out = F.Activation(conv_out, act_type='relu')
        if self.droprate > 0:
            conv_out = F.Dropout(conv_out, self.droprate)
        conv_out = self.conv2(conv_out)
        
        se_out = self.net_se_conv(conv_out)
        se_out = se_out.reshape(shape=se_out.shape + (1, 1,))
        return conv_out * se_out + residual


class CMPESEBlockV1(HybridBlock):
    '''
    Double FC
    '''
    
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(CMPESEBlockV1, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = _conv3x3(channels, stride, in_channels)
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv3x3(channels, 1, channels)
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False, in_channels=in_channels)
        else:
            self.downsample = None
        
        self.net_se_input_conv = nn.HybridSequential()
        self.net_se_input_conv.add(
            nn.GlobalAvgPool2D(),
            nn.Conv2D(channels=channels / net_se_ratio, kernel_size=1, use_bias=False),
            nn.Activation('relu')
        )
        self.net_se_input_x = nn.HybridSequential()
        self.net_se_input_x.add(
            nn.GlobalAvgPool2D(),
            nn.Conv2D(channels=channels / net_se_ratio, kernel_size=1, use_bias=False),
            nn.Activation('relu')
        )
        self.net_se = nn.HybridSequential()
        self.net_se.add(
            nn.Conv2D(channels=channels, kernel_size=1, use_bias=False),
            nn.Activation(activation='sigmoid')
        )
    
    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        conv_out = self.conv1(x)
        conv_out = self.bn2(conv_out)
        conv_out = F.Activation(conv_out, act_type='relu')
        conv_out = self.conv2(conv_out)
        
        se_input_conv = self.net_se_input_conv(conv_out)
        se_input_skipx = self.net_se_input_x(residual)
        se_input = nd.concat(se_input_conv, se_input_skipx, dim=1)
        se_out = self.net_se(se_input)
        return conv_out * se_out + residual


class CMPESEBlockV2(HybridBlock):
    '''
    Conv_2x1 or conv_1x1 pair-view
    '''
    
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(CMPESEBlockV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = _conv3x3(channels, stride, in_channels)
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv3x3(channels, 1, channels)
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False, in_channels=in_channels)
        else:
            self.downsample = None
        
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
            nn.Dense(channels / net_se_ratio, activation='relu', use_bias=False),
            nn.Dense(channels, activation='sigmoid', use_bias=False),
        )
    
    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        conv_out = self.conv1(x)
        conv_out = self.bn2(conv_out)
        conv_out = F.Activation(conv_out, act_type='relu')
        conv_out = self.conv2(conv_out)
        
        se_input_conv = self.net_Global_conv(conv_out)
        se_input_conv = se_input_conv.reshape(shape=(se_input_conv.shape[0], 1, se_input_conv.shape[1], 1))
        se_input_skipx = self.net_Global_skipx(residual)
        se_input_skipx = se_input_skipx.reshape(shape=(se_input_skipx.shape[0], 1, se_input_skipx.shape[1], 1))
        
        conv_x_concat = nd.concat(se_input_conv, se_input_skipx, dim=-1)
        multi_map = self.Multi_Map(conv_x_concat)
        one_map = nd.mean(multi_map, axis=1, keepdims=True)
        se_out = self.net_SE(one_map)
        se_out = se_out.reshape(shape=se_out.shape + (1, 1,))
        
        return conv_out * se_out + residual


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


CMPE_SE_block_versions = [CMPESEBlockV1, CMPESEBlockV2]


def _get_wrn_spec(num_layers, width_factor):
    assert (num_layers - 4) % 6 == 0
    n = (num_layers - 4) // 6
    layers = [n] * 3
    channels = [16, 16 * width_factor, 32 * width_factor, 64 * width_factor]
    return layers, channels


def get_se_wrn(num_layers, width_factor, **kwargs):
    layers, channels = _get_wrn_spec(num_layers, width_factor)
    net = CIFARWideResNet(SEBlock, layers, channels, **kwargs)
    return net


def get_cmpe_se_wrn(version, num_layers, width_factor, **kwargs):
    layers, channels = _get_wrn_spec(num_layers, width_factor)
    block_class = CMPE_SE_block_versions[version - 1]
    net = CIFARWideResNet(block_class, layers, channels, **kwargs)
    return net


def se_wrn28_10(**kwargs):
    return get_se_wrn(num_layers=28, width_factor=10, **kwargs)


def cmpe_se_v1_wrn28_10(**kwargs):
    return get_cmpe_se_wrn(version=1, num_layers=28, width_factor=10, **kwargs)


def cmpe_se_v2_wrn28_10(use_1x1=True, **kwargs):
    global CMPESEBlockV2_kernel
    CMPESEBlockV2_kernel = (1, 1) if use_1x1 else (1, 2)
    return get_cmpe_se_wrn(version=2, num_layers=28, width_factor=10, **kwargs)


# net = cmpe_se_v2_wrn28_10(use_1x1=True, classes=10)

def get_net():
    net = cmpe_se_v2_wrn28_10(use_1x1=True, classes=10)
    params_net = net.collect_params()
    params_net.initialize(ctx=ctx, init=init.Xavier())
    _ = net(nd.random_normal(shape=(128, 3, 32, 32), ctx=ctx))
    utils.print_params_num(params_net)
    return net, params_net


net, params_net = get_net()
