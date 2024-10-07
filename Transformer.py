# -*- coding: utf-8 -*-
'''
@File    :   Transformer.py
@Time    :   2024/10/07 19:40:57
@Author  :   Zhouqi Hua 
@Version :   1.0
@Desc    :   实现 transformer 的整体结构
'''

from torch import nn
import copy

from utils import WordEmbedding
from Attention import MHA
from FFN import FFN
from PositionEncoding import PositionEncoding
from Encoder import EncoderLayer, MH_Encoder
from Decoder import DecoderLayer, singleDecoder, biDecoder
from Generator import Generator

class Transformer_Zorolib(nn.Module):
    # 定义较小的层级结构、会变化的参数
    def __init__(self, vocab, dim_model, dim_ffn, n_heads, n_layers, dropout=0.1, device='cuda'):
        """
        - vocab: 词汇表
        - dim_model: 词向量（序列中每个元素）的维度 -> 基本层输入输出维度
        - dim_ffn: FFN 隐藏层维度。
            # 注：FNN模块的 __init__ 参数为 __init__(self, dim_model, dim_ffn, dropout=0.1)
        - n_heads: 多头注意力的头数
        - n_layers: Encoder/Decoder 层数
        """
        super(Transformer_Zorolib, self).__init__()
        self.vocab = vocab
        self.devide = device
        # 定义多头注意力计算层
        atten = MHA(n_heads, dim_model, dropout)
        # 定义前馈神经网络计算层
        feed_forward = FFN(dim_model, dim_ffn, dropout)
        
        # 输入嵌入层，词索引 -> 词向量
        self.src_embd = WordEmbedding(vocab.n_vocabs, dim_model)
        # 位置编码
        self.pos_embd = PositionEncoding(dim_model, dropout)
        
        # 初始化编码器
        self.encoder = MH_Encoder(EncoderLayer(dim_model, copy.deepcopy(atten), copy.deepcopy(feed_forward), dropout), n_layers)
        # 初始化单向解码器 (sublayer_num=3)
        self.single_decoder = singleDecoder(DecoderLayer(dim_model, copy.deepcopy(atten), copy.deepcopy(feed_forward), N=3, dropout=dropout), n_layers)
        # 初始化双向解码器 (sublayer_num=4)
        self.bi_decoder = biDecoder(DecoderLayer(dim_model, copy.deepcopy(atten), copy.deepcopy(feed_forward), N=4, dropout=dropout), n_layers)
        # 初始化生成器
        self.generator = Generator(dim_model, vocab.n_vocabs)
        
    # 编码操作函数
    def encode(self, src, mask):
        # 词嵌入
        src_embd = self.src_embd(src[0])
        # 位置编码
        src_pos = self.pos_embd(src_embd)
        # 编码
        return self.encoder(src_pos, mask[0])
    
    # 解码操作函数
    def sigle_decode(self, trg, encoder_output, mask):
        # 词嵌入
        trg_embd = self.src_embd(trg)
        # 位置编码
        trg_pos = self.pos_embd(trg_embd)
        # 解码
        return self.single_decoder(trg_pos, encoder_output, mask, mask)
        
    def forward(self, src, trg, mask):
        src_mask, r2l_padding_mask, r2l_trg_mask, trg_mask = mask
        enc_mask, dec_mask = src_mask
        # 编码
        encoder_output = self.encode(src, enc_mask)
        # 解码
        single_decoder_output = self.single_decode(trg, encoder_output, dec_mask)
        # 生成
        res = self.generator(single_decoder_output)
        return res
        
"""
TODO: 
1. Transformer_Zerolib 单向解码器改双向解码器
2. 数据集构建 + 完整的训练和测试代码
"""
        