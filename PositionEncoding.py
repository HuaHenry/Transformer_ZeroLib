# -*- coding: utf-8 -*-
'''
@File    :   PositionEncoding.py
@Time    :   2024/10/06 20:50:48
@Author  :   Zhouqi Hua 
@Version :   1.0
@Desc    :   实现余弦位置编码
'''

import math
import torch
import torch.nn as nn

"""
余弦位置编码类
"""
class PositionEncoding(nn.Module):
    def __init__(self, dim_model, dropout, max_len=5000):
        """
        - max_len：最大长度，一次性构建的位置编码的长度
        """
        super(PositionEncoding, self).__init__()
        # 偶数位置使用 sin 编码，奇数位置使用 cos 编码
        # 判断词向量是否为偶数维
        assert dim_model % 2 == 0
        # 位置编码矩阵
        PE = torch.zeros(max_len, dim_model)
        pos = torch.arange(0, max_len).unsqueeze(1)      # 位置索引（从0开始，对应位置编码公式中的position）
        div_GPU = torch.exp(torch.arange(0, dim_model, 2) * -(math.log(10000.0) / dim_model))
        """
        div_term：即 10000^{2i/d}
        - torch.arange(0, dim_model, 2)：创建偶数索引，生成一个从 0 ~ dim_model，步长为 2
        - -(math.log(10000.0) / dim_model)：计算 10000 的自然对数，然后将其除以 dim_model（缩放因子）
        并行生成完整的向量序列
        """
        PE[:, 0::2] = torch.sin(pos.float() * div_GPU)      # 0::2 表示从 0 开始，步长为 2
        PE[:, 1::2] = torch.cos(pos.float() * div_GPU)
        # 扩展维度
        PE = PE.unsqueeze(1)
        self.register_buffer('PE', PE)
        self.dropout = nn.Dropout(p=dropout)
        self.dim_model = dim_model
        
    def forward(self, x, step=None):
        """
        实现词向量和位置编码拼接
        - x：输入，实际上为 embedding 后的 x，维度表示为 [seq_len, batch_size, ...]
        - step：位置索引，用于实现动态位置编码（基础款transformer没有用到）
        """
        x = x * math.sqrt(self.dim_model)       # 乘以缩放因子
        if step is None:
            x = x + self.PE[:x.size(0), :]
        else:
            x = x + self.PE[step, :]
        return self.dropout(x)
        
        
    

