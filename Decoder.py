# -*- coding: utf-8 -*-
'''
@File    :   Decoder.py
@Time    :   2024/10/07 17:24:21
@Author  :   Zhouqi Hua 
@Version :   1.0
@Desc    :   实现 transformer 的解码器结构（包括单项解码器和双向解码器）
'''

from torch import nn

from ResNorm import ResNormBlock
from utils import cloneModule

"""
单头注意力解码器层
"""
class DecoderLayer(nn.Module):
    def __init__(self, size, atten, feed_forward, N, dropout=0.1):
        super(DecoderLayer, self).__init__()
        """
        - atten: 初始化的 MHA 层
        - N: 解码器内部子层数
        """
        self.atten = atten
        self.feed_forward = feed_forward
        self.sublayers = cloneModule(ResNormBlock(size, dropout), N)
        
    def forward(self, x, encoder_output, en_mask, de_mask, r2l_res=None, r2l_mask=None):
        """
        默认为单项解码器，不考虑r2l
        - x: 解码器输入
        - encoder_output: 编码器输出（作为解码器计算交叉注意力的 Q 和 K）
        - en_mask: 编码器输入掩码
        - de_mask: 解码器输入掩码（用于掩盖未来信息）
        
        若考虑双向解码器（即增加一个反向解码器输出的结果）
        - r2l_res: 反向解码器输出
        - r2l_mask: 反向解码器输入掩码
        """
        # 解码器自注意力
        first_res = self.sublayers[0](x, lambda first_atten: self.atten(x, x, x, de_mask))
        # 编码器-解码器交叉注意力
        second_res = self.sublayers[1](first_res, lambda second_atten: self.atten(first_res, encoder_output, encoder_output, en_mask))
        # 前馈神经网络（判断是否需要双向解码器）
        if r2l_res is None:
            return self.sublayers[-1](second_res, self.feed_forward)
        else:
            third_res = self.sublayers[-2](second_res, lambda third_atten: self.atten(second_res, r2l_res, r2l_res, r2l_mask))
            return self.sublayers[-1](third_res, self.feed_forward)
        
"""
单向（left-to-right）解码器
"""
class singleDecoder(nn.Module):
    def __init__(self, layer, N):
        super(singleDecoder, self).__init__()
        """
        - layer: 初始化的解码器层
        - N: Decoder层数
        """
        self.layers = cloneModule(layer, N)
        
    def forward(self, x, encoder_output, en_mask, de_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, en_mask, de_mask)
        return x
    
"""
双向（left-to-right & right-to-left）解码器
"""
class biDecoder(nn.Module):
    def __init__(self, layer, N):
        super(biDecoder, self).__init__()
        """
        - layer: 初始化的解码器层
        - N: Decoder层数
        """
        self.layers = cloneModule(layer, N)
        
    def forward(self, x, encoder_output, en_mask, de_mask, r2l_res, r2l_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, en_mask, de_mask, r2l_res, r2l_mask)
        return x