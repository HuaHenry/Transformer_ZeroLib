# -*- coding: utf-8 -*-
'''
@File    :   Attention.py
@Time    :   2024/10/06 20:46:29
@Author  :   Zhouqi Hua 
@Version :   1.0
@Desc    :   手写实现自注意力计算；实现多头注意力模块
'''

import torch
import torch.nn as nn
import math

"""
自注意力计算函数
"""
def self_attention(query, key, value, mask=None, dropout=None):
    # 计算 q 和 k 的相似度得分
    # - torch.matmul：矩阵相乘
    # - transpose(-2, -1)：对最后两个维度进行转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / \
        math.sqrt(query.size(-1))
    # mask 掩码
    if mask is not None:
        mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)
    # 计算掩码后的 self-attention
    self_atten = nn.functional.softmax(scores, dim=-1)
    if dropout is not None:
        self_atten = dropout(self_atten)
    # 自注意力得分计算（返回加权和和 self-attention）
    return torch.matmul(self_atten, value), self_atten


"""
多头注意力机制类（Multi-Head Attention）
- head_num：多头注意力的头数
- dim_model：输入维度
"""
class MHA(nn.Module):
    def __init__(self, head_num, dim_model, dropout=0.1):
        super(MHA, self).__init__()
        # 使用断言确保向量维度是头数的整数倍（才可以平均分为多头）
        assert not (dim_model % head_num)
        # 定义参数
        self.dim_perhead = dim_model // head_num       # 单头维度
        self.head_num = head_num                    # 头数
        self.dim_model = dim_model                  # 输入维度
        self.dropout = nn.Dropout(p=dropout)        # 丢弃率
        self.atten_softmax = nn.Softmax(dim=-1)     # 相似度量值，softmax(QK^T)
        # 针对同一输入 x，实现同源线性变换，得到 Q/K/V
        self.linear_q = nn.Linear(dim_model, dim_model)
        self.linear_k = nn.Linear(dim_model, dim_model)
        self.linear_v = nn.Linear(dim_model, dim_model)
        # 最后的线性变换
        self.linear_f = nn.Linear(dim_model, dim_model)

    def forward(self, q, k, v, mask=None):
        """
        q/k/v 原始输入为相同的 x
        x 的维度为 [n_batch, seq_len, dim_model]
        - n_batch：批次大小
        - seq_len：序列长度
        - dim_model：序列每一个元素的维度
        """
        if mask is not None:
            mask.unsqueeze(1)                      # 在第1维增加一个维度
        n_batches = q.size(0)                      # 批次大小（第一维存放batch）
        # 线性变换得到 Q/K/V
        # - view：改变张量形状，从 [n_batch, seq_len, dim_model] 变为 [n_batch, seq_len, head_num, dim_perhead]
        # 多头切分的核心：dim_model -> head_num * dim_perhead
        # - 参数 -1：表示自动计算该维度大小，使得总元素个数不变
        # - transpose(1,2)：交换第2/3维，即转化为为 [n_batch, head_num, seq_len, dim_perhead]
        q = self.linear_q(q).view(n_batches, -1, self.head_num,
                                  self.dim_perhead).transpose(1, 2)
        k = self.linear_k(k).view(n_batches, -1, self.head_num,
                                  self.dim_perhead).transpose(1, 2)
        v = self.linear_v(v).view(n_batches, -1, self.head_num,
                                  self.dim_perhead).transpose(1, 2)
        # 调用 self_attention 函数计算多头注意力
        x, self_atten = self_attention(q, k, v, mask, self.dropout, mask)
        # 维度恢复：[n_batch, head_num, seq_len, dim_perhead] -> [n_batch, seq_len, dim_model]
        x = x.transpose(1, 2).contiguous().view(
            n_batches, -1, self.head_num * self.dim_perhead)
        # 最后层线性变换
        return self.linear_f(x)

