import torch.nn as nn
from Function.Function import *
from training.PositionalEncoding import*
from training.Switch import *

class EncoderLayer(nn.Module):
    def __init__(self,d_model,h,ExpertNum,local_ExpertNum,GpuNum,dropout,ProcessId,sigma,group,lb_coef):
        super().__init__()
        # —————————————— # 自注意力
        self.MhaLayerNorm = nn_LayerNorm(d_model)
        self.SelfMha = nn.MultiheadAttention(embed_dim=d_model,num_heads=h,dropout=dropout,batch_first=True)
        # —————————————— # 全连接
        self.FFNLayerNorm = nn_LayerNorm(d_model)
        self.Switch = Switch(d_model,ExpertNum,local_ExpertNum,GpuNum,ProcessId,sigma,group,lb_coef)
        # —————————————— # dropout
        self.dropout = nn.Dropout(dropout)
    def forward(self,src,src_pad_mask,experts,B,T,D,N,idx_all,N_all,cap,dtype_mid,local_rank_mid,rand_all):
        # —————————————— # 自注意力
        src_ = self.MhaLayerNorm(src)
        src_ = self.SelfMha(src_,src_,src_,key_padding_mask=src_pad_mask,need_weights=False)[0]
        src  = src + self.dropout(src_)
        # —————————————— # 全连接
        src_ = self.FFNLayerNorm(src)
        src_,loss = self.Switch(src_,experts,B,T,D,N,idx_all,N_all,cap,dtype_mid,local_rank_mid,rand_all)
        src = src + self.dropout(src_)
        return src,loss

class Encoder(nn.Module):
    def __init__ (self,d_model,h,N,vocab_size,ExpertNum,local_ExpertNum,GpuNum,dropout,ProcessId,capacity_factor,sigma,group,lb_coef):
        super().__init__()
        # —————————————— # 实例化
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.positionalencoding = PositionalEncoding(d_model, dropout)
        self.layers  = nn.ModuleList([EncoderLayer(d_model,h,ExpertNum,local_ExpertNum,GpuNum,dropout,ProcessId,sigma,group,lb_coef) for _ in range(N)])
        self.EndLayerNorm = nn_LayerNorm(d_model)
        # —————————————— # 初始化参数
        self.scale = d_model**0.5
        self.capacity_factor = capacity_factor
        self.ExpertNum = ExpertNum
    def forward(self,src,src_pad_mask,expertset):
        # —————————————— # 词嵌入 & 位置嵌入
        src = self.positionalencoding(self.tok_embedding(src)*self.scale) # 词嵌入
        
        # —————————————— # 制作mask_all & idx_all
        B,T,D  = src.shape ; N = B * T
        mask_all = src_pad_mask.reshape(N)
        idx_all  = torch.nonzero(~mask_all, as_tuple=False).squeeze(1)
        N_all = int(idx_all.numel())
        cap = min(N_all,int((N_all*self.capacity_factor)/self.ExpertNum))
        del mask_all

        # —————————————— # 制作 tensor
        dtype_mid = torch.get_autocast_dtype('cuda')
        local_rank_mid = torch.arange(N_all, device='cuda')
        rand_all = torch.randperm(N_all, device='cuda') 
        LossSum = 0.0

        # —————————————— # 前向传播
        for layer,expert in zip(self.layers,expertset):
            src,LossUnit = layer(src,src_pad_mask,expert
                                ,B,T,D,N,idx_all,N_all,cap 
                                ,dtype_mid,local_rank_mid,rand_all)
            LossSum = LossSum + LossUnit
        
        # —————————————— # 输出
        src = self.EndLayerNorm(src)
        return src,LossSum