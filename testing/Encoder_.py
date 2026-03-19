import torch.nn as nn
from Function.Function import *
from testing.PositionalEncoding_ import*
from testing.Switch_ import *

class EncoderLayer(nn.Module):
    def __init__(self,d_model,h,ExpertNum):
        super(EncoderLayer, self).__init__()
        # —————————————— # 自注意力
        self.MhaLayerNorm = nn_LayerNorm(d_model)
        self.SelfMha = nn.MultiheadAttention(embed_dim=d_model,num_heads=h,batch_first=True)
        # —————————————— # 全连接
        self.FFNLayerNorm = nn_LayerNorm(d_model)
        self.Switch = Switch(d_model,ExpertNum)
    def forward(self,src,src_pad_mask,token_mask,experts):
        # —————————————— # 自注意力
        src_ = self.MhaLayerNorm(src)
        src_ = self.SelfMha(src_,src_,src_,key_padding_mask=src_pad_mask,need_weights=False)[0]
        src += src_
        # —————————————— # 全连接
        src_ = self.FFNLayerNorm(src)
        src_ = self.Switch(src_,experts,token_mask)
        src += src_
        return src

class Encoder(nn.Module):
    def __init__ (self,d_model,h,N,vocab_size,ExpertNum):
        super(Encoder, self).__init__()
        # —————————————— # 实例化
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.positionalencoding = PositionalEncoding(d_model)
        self.layers  = nn.ModuleList([EncoderLayer(d_model,h,ExpertNum) for _ in range(N)])
        self.EndLayerNorm = nn_LayerNorm(d_model)
        # —————————————— # 初始化参数
        self.N = N
        self.scale = d_model**0.5
    def forward(self,src,src_pad_mask,expertsset):
        # —————————————— # 初始化
        src = self.positionalencoding(self.tok_embedding(src)*self.scale) # 词嵌入
        token_mask = ~src_pad_mask        # mask反转
        # —————————————— # 前向传播
        for layer,experts in zip(self.layers,expertsset):
            src = layer(src,src_pad_mask,token_mask,experts)
        # —————————————— # 输出
        src = self.EndLayerNorm(src)
        return src