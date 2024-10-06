# -*- coding: utf-8 -*-
'''
@File    :   Mask.py
@Time    :   2024/10/06 23:07:29
@Author  :   Zhouqi Hua 
@Version :   1.0
@Desc    :   实现 decoder 的掩码
'''

import torch
import numpy as np

"""
生成上三角的 bool 型掩码矩阵，使其不关注未来信息
"""
def upper_triangular_mask(dim):
    mask = np.triu(np.ones((1,dim,dim)),k=1).astype('uint8')
    """
    - np.ones((1, dim, dim))：生成全 1 矩阵
    - np.triu()：返回矩阵的上三角部分
    - k=1：指定对角线的位置（从第一条副对角线开始保留元素，即对角线以上的元素将被保留，而对角线及以下的元素将被置零）
    - astype('uint8')：转换为 uint8 类型（减少内存占用）
    """
    return (torch.from_numpy(mask) == 0).cuda






