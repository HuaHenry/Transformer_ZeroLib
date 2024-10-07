# -*- coding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/10/06 15:02:08
@Author  :   Zhouqi Hua 
@Version :   1.0
@Desc    :   工具函数（包含一些基本层和操作的手写实现）
'''

import torch
import torch.nn as nn
import copy

"""
层归一化函数，用于构建Transformer的中的LN结构
"""
class LayerNorm:
    def __init__(self, size, eps=1e-6):
        # size 表示输入的特征维度（即self-attention的x的大小）
        self.gama = nn.Parameter(torch.ones(size))
        self.beta = nn.Parameter(torch.zeros(size))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # -1：平均在最后一个维度上
        std = x.std(-1, keepdim=True)
        return self.gama * (x - mean) / (std + self.eps) + self.beta 
    
"""
克隆模块函数
作用：
- 用于构建多头注意力机制中的多个注意力头
- 克隆encoder、decoder中的多个残差链接+归一化结构
克隆到 nn.ModuleList 中的 module 会被自动注册到模型的参数中
"""
def cloneModule(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])