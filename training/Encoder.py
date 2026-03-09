import torch.nn as nn
from Function.Function import *
from training.PositionalEncoding import*
from training.Switch import *

class EncoderLayer(nn.Module):
    def __init__(self,d_model,h,ExpertNum,local_ExpertNum,GpuNum,dropout,ProcessId,capacity_factor,sigma,group):
        super(EncoderLayer, self).__init__()
        # —————————————— # 自注意力
        self.MhaLayerNorm = nn_LayerNorm(d_model)
        self.SelfMha = nn.MultiheadAttention(embed_dim=d_model,num_heads=h,dropout=dropout,batch_first=True)
        # —————————————— # 全连接
        self.FFNLayerNorm = nn_LayerNorm(d_model)
        self.Switch = Switch(d_model,ExpertNum,local_ExpertNum,GpuNum,ProcessId,capacity_factor,sigma,group)
        # —————————————— # dropout
        self.dropout = nn.Dropout(dropout)
    def forward(self,src,src_pad_mask,token_mask,experts):
        # —————————————— # 自注意力
        src_ = self.MhaLayerNorm(src)
        src_ = self.SelfMha(src_,src_,src_,key_padding_mask=src_pad_mask,need_weights=False)[0]
        src  = src + self.dropout(src_)
        # —————————————— # 全连接
        src_ = self.FFNLayerNorm(src)
        src_,lb_loss,z_loss = self.Switch(src_, experts, token_mask=token_mask)
        src = src + self.dropout(src_)
        return src,lb_loss,z_loss

class Encoder(nn.Module):
    def __init__ (self,d_model,h,N,vocab_size,ExpertNum,local_ExpertNum,GpuNum,dropout,ProcessId,capacity_factor,sigma,group):
        super(Encoder, self).__init__()
        # —————————————— # 实例化
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.positionalencoding = PositionalEncoding(d_model, dropout)
        self.layers  = nn.ModuleList([EncoderLayer(d_model,h,ExpertNum,local_ExpertNum,GpuNum,dropout,ProcessId,capacity_factor,sigma,group) for _ in range(N)])
        self.EndLayerNorm = nn_LayerNorm(d_model)
        # —————————————— # 初始化参数
        self.N = N
        self.scale = d_model**0.5
    def forward(self,src,src_pad_mask,expertsset):
        # —————————————— # 初始化
        src = self.positionalencoding(self.tok_embedding(src)*self.scale) # 词嵌入
        lb_loss_sum ,z_loss_sum= 0.0,0.0  # 损失初始化
        token_mask = ~src_pad_mask        # mask反转
        # —————————————— # 前向传播
        for layer,experts in zip(self.layers,expertsset):
            src,lb_loss,z_loss = layer(src,src_pad_mask,token_mask,experts)
            lb_loss_sum = lb_loss_sum + lb_loss
            z_loss_sum  = z_loss_sum  + z_loss
        # —————————————— # 输出
        src = self.EndLayerNorm(src)
        return src, lb_loss_sum/self.N, z_loss_sum/self.N