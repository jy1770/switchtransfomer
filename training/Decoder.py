import torch.nn as nn
from Function.Function import *
from training.PositionalEncoding import*
from training.Switch import *

class DecoderLayer(nn.Module):
    def __init__(self,d_model,h,ExpertNum,local_ExpertNum,GpuNum,dropout,ProcessId,capacity_factor,sigma,group):
        super(DecoderLayer, self).__init__()
        # —————————————— # 自注意力
        self.SelfMhaLayerNorm = nn_LayerNorm(d_model)
        self.SelfMha = nn.MultiheadAttention(embed_dim=d_model, num_heads=h, dropout=dropout, batch_first=True)
        # —————————————— # 混合注意力
        self.EncMhaLayerNorm = nn_LayerNorm(d_model)
        self.EncMha = nn.MultiheadAttention(embed_dim=d_model, num_heads=h, dropout=dropout, batch_first=True)
        # —————————————— # 全连接
        self.FFNLayerNorm = nn_LayerNorm(d_model)
        self.Switch = Switch(d_model,ExpertNum,local_ExpertNum,GpuNum,ProcessId,capacity_factor,sigma,group)
        # —————————————— # dropout
        self.dropout = nn.Dropout(dropout)
    def forward(self,tgt,src,tgt_mask,tgt_pad_mask,src_pad_mask,token_mask,experts):
        # —————————————— # 自注意力
        tgt_ = self.SelfMhaLayerNorm(tgt)
        tgt_ = self.SelfMha(tgt_,tgt_,tgt_,key_padding_mask = tgt_pad_mask, attn_mask = tgt_mask,need_weights=False)[0]
        tgt  = tgt + self.dropout(tgt_)
        # —————————————— # 混合注意力
        tgt_ = self.EncMhaLayerNorm(tgt)
        tgt_ = self.EncMha(tgt_,src,src,key_padding_mask=src_pad_mask,need_weights=False)[0]
        tgt  = tgt + self.dropout(tgt_)
        # —————————————— # 全连接
        tgt_ = self.FFNLayerNorm(tgt)
        tgt_,lb_loss,z_loss = self.Switch(tgt_, experts,token_mask)
        tgt  = tgt + self.dropout(tgt_)
        return tgt,lb_loss,z_loss

class Decoder(nn.Module):
    def __init__(self,d_model,h,N,vocab_size,ExpertNum,local_ExpertNum,GpuNum,dropout,ProcessId,capacity_factor,sigma,group):
        super(Decoder, self).__init__()
        # —————————————— # 实例化
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.positionalencoding = PositionalEncoding(d_model, dropout)
        self.layers  = nn.ModuleList([DecoderLayer(d_model,h,ExpertNum,local_ExpertNum,GpuNum,dropout,ProcessId,capacity_factor,sigma,group) for _ in range(N)])
        self.fc_out  = nn.Linear(d_model, vocab_size)
        self.EndLayerNorm = nn_LayerNorm(d_model)
        # —————————————— # 初始化参数
        self.N = N
        self.scale = d_model**0.5
    def forward(self,tgt,src,tgt_mask,tgt_pad_mask,src_pad_mask,expertsset):
        # —————————————— # 嵌入
        tgt = self.positionalencoding(self.tok_embedding(tgt)*self.scale)
        lb_loss_sum,z_loss_sum = 0.0,0.0 
        token_mask = ~tgt_pad_mask
        # —————————————— # 前向传播
        for layer,experts in zip(self.layers,expertsset):
            tgt,lb_loss,z_loss = layer(tgt,src,tgt_mask,tgt_pad_mask,src_pad_mask,token_mask,experts)
            lb_loss_sum = lb_loss_sum + lb_loss# lb_loss是torch
            z_loss_sum = z_loss_sum + z_loss   # z_loss是torch
        # —————————————— # 合成输出
        tgt = self.EndLayerNorm(tgt)
        output = self.fc_out(tgt)
        return output,lb_loss_sum/self.N,z_loss_sum/self.N