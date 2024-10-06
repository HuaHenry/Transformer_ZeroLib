# -*- coding: utf-8 -*-
'''
@File    :   Generator.py
@Time    :   2024/10/06 22:41:33
@Author  :   Zhouqi Hua 
@Version :   1.0
@Desc    :   Linear + Softmax 层实现生成器
'''

import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, dim_model, vocab_size):
        super(Generator, self).__init__()
        # 线性变换
        self.proj = nn.Linear(dim_model, vocab_size)

    def forward(self, x):
        return nn.functional.log_softmax(self.proj(x), dim=-1)

