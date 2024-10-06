# -*- coding: utf-8 -*-
'''
@File    :   components.py
@Time    :   2024/10/06 16:26:39
@Author  :   Zhouqi Hua 
@Version :   1.0
@Desc    :   Transformer的组成部件
'''

import torch.nn as nn

# 调用工具函数文件 utils.py
from utils import LayerNorm

"""
残差块和层归一化（LN）块的结合，用于构建Transformer的中的ADD+LN结构
"""
class ResNormBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(ResNormBlock, self).__init__()
        # 首先实现 LayerNorm
        self.norm = LayerNorm(dim)
        # 接着实现 dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        # x + sublayer(x) 实现残差连接
        return self.dropout(self.layer_norm(x + sublayer(x)))