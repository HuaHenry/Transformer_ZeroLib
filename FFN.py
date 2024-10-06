# -*- coding: utf-8 -*-
'''
@File    :   FFN.py
@Time    :   2024/10/06 22:30:34
@Author  :   Zhouqi Hua 
@Version :   1.0
@Desc    :   实现前馈神经网络 Feed Forward Network（FFN）模块
'''

import torch.nn as nn

"""
posion-wise feed forward network模块
核心：w2(relu(w1(layer_norm(x))+b1))+b2
参数设置时b1/b2可以不设置，直接在w1/w2中设置偏置
"""
class FFN(nn.Module):
    def __init__(self, dim_model, dim_ffn, dropout=0.1):
        super(FFN, self).__init__()
        # 线性变换
        self.linear1 = nn.Linear(dim_model, dim_ffn)
        self.linear2 = nn.Linear(dim_ffn, dim_model)
        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)    # 也可以区分两个dropout
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 内层线性变换
        x = self.dropout(self.relu(self.linear1(self.layer_norm(x))))
        # 外层线性变换
        x = self.dropout(self.linear2(x))
        return x


