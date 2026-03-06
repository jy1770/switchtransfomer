import torch
import torch.nn as nn
from Function.Function import *
from testing.PositionalEncoding_ import*
from testing.Switch_ import *

class DecoderLayer(nn.Module):
    def __init__(self,d_model,h,ExpertNum):
        super(DecoderLayer, self).__init__()
        # —————————————— # 自注意力
        self.SelfMhaLayerNorm = nn_LayerNorm(d_model)
        self.SelfMha = nn.MultiheadAttention(embed_dim=d_model, num_heads=h, batch_first=True)
        # —————————————— # 混合注意力
        self.EncMhaLayerNorm_src = nn_LayerNorm(d_model)
        self.EncMhaLayerNorm_tgt = nn_LayerNorm(d_model)
        self.EncMha = nn.MultiheadAttention(embed_dim=d_model, num_heads=h, batch_first=True)
        # —————————————— # 全连接
        self.FFNLayerNorm = nn_LayerNorm(d_model)
        self.Switch = Switch(d_model,ExpertNum)
    def forward(self,tgt,src,tgt_mask,tgt_pad_mask,src_pad_mask,experts):
        # —————————————— # 自注意力
        tgt_ = self.SelfMhaLayerNorm(tgt)
        tgt_ = self.SelfMha(tgt_,tgt_,tgt_,key_padding_mask = tgt_pad_mask, attn_mask = tgt_mask,need_weights=False)[0]
        tgt  = tgt + tgt_
        # —————————————— # 混合注意力
        src_ = self.EncMhaLayerNorm_src(src)
        tgt_ = self.EncMhaLayerNorm_tgt(tgt)
        tgt_ = self.EncMha(tgt_,src_,src_,key_padding_mask=src_pad_mask,need_weights=False)[0]
        tgt  = tgt + tgt_
        # —————————————— # 全连接
        token_mask = ~tgt_pad_mask
        tgt_ = self.FFNLayerNorm(tgt)
        tgt_ = self.Switch(tgt_,experts,token_mask)
        tgt = tgt + tgt_
        return tgt


class Decoder(nn.Module):
    def __init__(self,d_model,h,N,vocab_size,ExpertNum):
        super(Decoder, self).__init__()
        # —————————————— # 实例化
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.positionalencoding = PositionalEncoding(d_model)
        self.layers  = nn.ModuleList([DecoderLayer(d_model,h,ExpertNum) for _ in range(N)])
        self.fc_out  = nn.Linear(d_model , vocab_size)
        self.EndLayerNorm = nn_LayerNorm(d_model)
        # —————————————— # 初始化参数
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(f'cuda:0')
    def forward(self,tgt,src,tgt_mask,tgt_pad_mask,src_pad_mask,expertsset):
        # —————————————— # 嵌入
        tgt = self.positionalencoding(self.tok_embedding(tgt)*self.scale)
        # —————————————— # 前向传播
        for i,layer in enumerate(self.layers):
            tgt  = layer(tgt,src,tgt_mask,tgt_pad_mask,src_pad_mask,expertsset[i])
        # —————————————— # 合成输出
        tgt = self.EndLayerNorm(tgt)
        output = self.fc_out(tgt)
        return output