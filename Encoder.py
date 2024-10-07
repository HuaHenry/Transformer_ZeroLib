# -*- coding: utf-8 -*-
'''
@File    :   Encoder.py
@Time    :   2024/10/07 16:45:02
@Author  :   Zhouqi Hua 
@Version :   1.0
@Desc    :   实现 transformer 的编码器结构
'''

from torch import nn

from ResNorm import ResNormBlock
from utils import cloneModule

"""
单头注意力编码器层
"""
class EncoderLayer(nn.module):
    def __init__(self, size, self_atten, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_atten = self_atten                                    # 初始化的 MHA 层
        self.feed_forward = feed_forward                                # 初始化的 FFN 层
        self.sublayers = cloneModule(ResNormBlock(size, dropout), 2)     # 克隆两个残差连接+归一化结构
        
    def forward(self, x, mask):
        # 残差连接+归一化结构
        x = self.sublayers[0](x, lambda x: self.self_atten(x, x, x, mask))
        res = self.sublayers[1](x, self.feed_forward)
        
"""
多头注意力编码器
"""
class MH_Encoder(nn.Module):
    def __init__(self, layer, N):
        super(MH_Encoder, self).__init__()
        # 定义 N 个单头注意力编码器层的集合 -> 用于构建多头注意力编码器
        self.MH_layers = cloneModule(layer, N)      # Attention is all you need 中 N=6
        
    def forward(self, x, mask):
        for layer in self.MH_layers:
            x = layer(x, mask)      # 掩码（实际在Encoder中不需要）
        return x
