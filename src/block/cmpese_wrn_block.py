# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import ndarray as nd

net_se_ratio = 16
cmpe_se_ratio = 16
reimage_k = 10  # width_factor wrn28-10

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
        shortcut = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            shortcut = self.downsample(x)
        conv_out = self.conv1(x)
        conv_out = self.bn2(conv_out)
        conv_out = F.Activation(conv_out, act_type='relu')
        if self.droprate > 0:
            conv_out = F.Dropout(conv_out, self.droprate)
        conv_out = self.conv2(conv_out)
        
        se_out = self.net_se_conv(conv_out)
        se_out = se_out.reshape(shape=se_out.shape + (1, 1,))
        return conv_out * se_out + shortcut


class CMPESEBlockDoubleFC(HybridBlock):
    '''
    Double FC
    '''
    
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(CMPESEBlockDoubleFC, self).__init__(**kwargs)
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
        shortcut = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            shortcut = self.downsample(x)
        conv_out = self.conv1(x)
        conv_out = self.bn2(conv_out)
        conv_out = F.Activation(conv_out, act_type='relu')
        conv_out = self.conv2(conv_out)
        
        se_input_conv = self.net_se_input_conv(conv_out)
        se_input_skipx = self.net_se_input_x(shortcut)
        se_input = nd.concat(se_input_conv, se_input_skipx, dim=1)
        se_out = self.net_se(se_input)
        return conv_out * se_out + shortcut


class CMPESEBlock1x1(HybridBlock):
    '''
    Conv_1x1 pair-view
    '''
    
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(CMPESEBlock1x1, self).__init__(**kwargs)
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
            nn.Conv2D(channels=channels / cmpe_se_ratio, kernel_size=(1, 1), use_bias=False),
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
        shortcut = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            shortcut = self.downsample(x)
        conv_out = self.conv1(x)
        conv_out = self.bn2(conv_out)
        conv_out = F.Activation(conv_out, act_type='relu')
        conv_out = self.conv2(conv_out)
        
        se_input_conv = self.net_Global_conv(conv_out)
        se_input_conv = se_input_conv.reshape(shape=(se_input_conv.shape[0], 1, se_input_conv.shape[1], 1))
        se_input_skipx = self.net_Global_skipx(shortcut)
        se_input_skipx = se_input_skipx.reshape(shape=(se_input_skipx.shape[0], 1, se_input_skipx.shape[1], 1))
        
        conv_x_concat = nd.concat(se_input_conv, se_input_skipx, dim=-1)
        multi_map = self.Multi_Map(conv_x_concat)
        one_map = nd.mean(multi_map, axis=1, keepdims=True)
        se_out = self.net_SE(one_map)
        se_out = se_out.reshape(shape=se_out.shape + (1, 1,))
        
        return conv_out * se_out + shortcut


class CMPESEBlock2x1(HybridBlock):
    '''
    Conv_2x1 pair-view
    '''
    
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(CMPESEBlock2x1, self).__init__(**kwargs)
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
            nn.Conv2D(channels=channels / cmpe_se_ratio, kernel_size=(1, 2), use_bias=False),
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
        shortcut = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            shortcut = self.downsample(x)
        conv_out = self.conv1(x)
        conv_out = self.bn2(conv_out)
        conv_out = F.Activation(conv_out, act_type='relu')
        conv_out = self.conv2(conv_out)
        
        se_input_conv = self.net_Global_conv(conv_out)
        se_input_conv = se_input_conv.reshape(shape=(se_input_conv.shape[0], 1, se_input_conv.shape[1], 1))
        se_input_skipx = self.net_Global_skipx(shortcut)
        se_input_skipx = se_input_skipx.reshape(shape=(se_input_skipx.shape[0], 1, se_input_skipx.shape[1], 1))
        
        conv_x_concat = nd.concat(se_input_conv, se_input_skipx, dim=-1)
        multi_map = self.Multi_Map(conv_x_concat)
        one_map = nd.mean(multi_map, axis=1, keepdims=True)
        se_out = self.net_SE(one_map)
        se_out = se_out.reshape(shape=se_out.shape + (1, 1,))
        
        return conv_out * se_out + shortcut


class WRNFoldReimageLayer(HybridBlock):
    def __init__(self, group=reimage_k, **kwargs):
        super(WRNFoldReimageLayer, self).__init__(**kwargs)
        self._group = group
    
    def hybrid_forward(self, F, x):  # x(N, 1, C, 2)
        # (n,1,c,2)->(n,k,c/k,2)
        concat_step1 = x.reshape((x.shape[0], self._group, x.shape[2] / self._group, 2))
        # (n,k,c/k,2)->(n,k,2,c/k)
        concat_step2 = nd.transpose(concat_step1, (0, 1, 3, 2))
        # (n,k,2,c/k)->(n, 1, 2*k,c/k)
        concat_out = concat_step2.reshape((concat_step2.shape[0], 1, 2 * self._group, concat_step2.shape[-1]))
        return concat_out


class CMPESEBlock3x3(HybridBlock):
    def __init__(self, channels, stride, downsample, in_channels=0, **kwargs):
        super(CMPESEBlock3x3, self).__init__(**kwargs)
        self.channels = channels
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
        
        self.net_reimage_layer = nn.HybridSequential()
        self.net_reimage_layer.add(
            WRNFoldReimageLayer(group=reimage_k)
        )
        
        self.Multi_Map = nn.HybridSequential()
        self.Multi_Map.add(
            nn.Conv2D(channels=channels / cmpe_se_ratio, kernel_size=(3, 3), use_bias=False),
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
        
        se_input_conv = self.net_Global_conv(conv_out)  # (N, C, 1, 1)
        se_input_conv = se_input_conv.reshape(shape=(se_input_conv.shape[0], 1, se_input_conv.shape[1], 1))
        
        se_input_skipx = self.net_Global_skipx(residual)  # (N, C, 1, 1)
        se_input_skipx = se_input_skipx.reshape(shape=(se_input_skipx.shape[0], 1, se_input_skipx.shape[1], 1))
        
        conv_x_concat = nd.concat(se_input_conv, se_input_skipx, dim=-1)  # (N, 1, C, 2)
        conv_x_concat_out = self.net_reimage_layer(conv_x_concat)  # (n, 1, 2*k,c/k)
        
        multi_map = self.Multi_Map(conv_x_concat_out)  # (N, m, 2*k-2, c/k-2)
        one_map = nd.mean(multi_map, axis=1, keepdims=True)  # (N, 1, 2*k-2, c/k-2)
        
        se_out = self.net_SE(one_map)  # (N, C)
        se_out = se_out.reshape(shape=se_out.shape + (1, 1,))  # (N, C , 1, 1)
        
        return conv_out * se_out + residual
