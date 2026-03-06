import torch
import torch.nn as nn
from Function.Function import *
from testing.PositionalEncoding_ import*
from testing.Switch_ import *

class EncoderLayer(nn.Module):
    def __init__(self,d_model,h,ExpertNum):
        super(EncoderLayer, self).__init__()
        # —————————————— # 自注意力
        self.SelfMha = nn.MultiheadAttention(embed_dim=d_model,num_heads=h,batch_first=True)
        self.MhaLayerNorm = nn_LayerNorm(d_model)\
        # —————————————— # 全连接
        self.Switch = Switch(d_model,ExpertNum)
        self.FFNLayerNorm = nn_LayerNorm(d_model)

    def forward(self,src,src_pad_mask,experts):
        # —————————————— # 自注意力
        src_ = self.MhaLayerNorm(src)
        src_ = self.SelfMha(src_,src_,src_,key_padding_mask=src_pad_mask,need_weights=False)[0]
        src  = src + src_
        # —————————————— # 全连接
        token_mask = ~src_pad_mask
        src_ = self.FFNLayerNorm(src)
        src_ = self.Switch(src_,experts,token_mask)
        src = src + src_
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
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(f'cuda:0')
    def forward(self,src,src_pad_mask,expertsset):
        # —————————————— # 嵌入
        src = self.positionalencoding(self.tok_embedding(src)*self.scale)
        # —————————————— # 前向传播
        for i,layer in enumerate(self.layers):
            src = layer(src,src_pad_mask,expertsset[i])
        # —————————————— # 输出
        src = self.EndLayerNorm(src)
        return src





