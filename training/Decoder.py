import torch.nn as nn
from Function.Function import *
from training.PositionalEncoding import*
from training.Switch import *

class DecoderLayer(nn.Module):
    def __init__(self,d_model,h,ExpertNum,local_ExpertNum,GpuNum,dropout,ProcessId,sigma,group,lb_coef,z_coef):
        super().__init__()
        # —————————————— # 自注意力
        self.SelfMhaLayerNorm = nn_LayerNorm(d_model)
        self.SelfMha = nn.MultiheadAttention(embed_dim=d_model, num_heads=h, dropout=dropout, batch_first=True)
        # —————————————— # 混合注意力
        self.EncMhaLayerNorm = nn_LayerNorm(d_model)
        self.EncMha = nn.MultiheadAttention(embed_dim=d_model, num_heads=h, dropout=dropout, batch_first=True)
        # —————————————— # 全连接
        self.FFNLayerNorm = nn_LayerNorm(d_model)
        self.Switch = Switch(d_model,ExpertNum,local_ExpertNum,GpuNum,ProcessId,sigma,group,lb_coef,z_coef)
        # —————————————— # dropout
        self.dropout = nn.Dropout(dropout)
    def forward(self,tgt,src,tgt_mask,tgt_pad_mask,src_pad_mask,experts,B,T,D,N,idx_all,N_all,cap,dtype_mid,local_rank_mid,rand_all):
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
        tgt_,loss = self.Switch(tgt_,experts,B,T,D,N,idx_all,N_all,cap,dtype_mid,local_rank_mid,rand_all)
        tgt  = tgt + self.dropout(tgt_)
        return tgt,loss

class Decoder(nn.Module):
    def __init__(self,d_model,h,N,vocab_size,ExpertNum,local_ExpertNum,GpuNum,dropout,ProcessId,capacity_factor,sigma,group,lb_coef,z_coef):
        super().__init__()
        # —————————————— # 实例化
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.positionalencoding = PositionalEncoding(d_model, dropout)
        self.layers  = nn.ModuleList([DecoderLayer(d_model,h,ExpertNum,local_ExpertNum,GpuNum,dropout,ProcessId,sigma,group,lb_coef,z_coef) for _ in range(N)])
        self.fc_out  = nn.Linear(d_model, vocab_size)
        self.EndLayerNorm = nn_LayerNorm(d_model)
        # —————————————— # 初始化参数
        self.scale = d_model**0.5
        self.capacity_factor = capacity_factor
        self.ExpertNum = ExpertNum
    def forward(self,tgt,src,tgt_mask,tgt_pad_mask,src_pad_mask,expertset):
        # —————————————— # 词嵌入 & 位置嵌入
        tgt = self.positionalencoding(self.tok_embedding(tgt)*self.scale)
        
        # —————————————— # 制作mask_all & idx_all
        B,T,D  = tgt.shape ; N = B * T
        mask_all = tgt_pad_mask.reshape(N)
        idx_all  = torch.nonzero(~mask_all, as_tuple=False).squeeze(1)
        N_all = int(idx_all.numel())
        cap = min(N_all , int((N_all*self.capacity_factor)/self.ExpertNum))
        del mask_all

        # —————————————— # 制作 tensor
        dtype_mid = torch.get_autocast_dtype('cuda')
        local_rank_mid = torch.arange(N_all, device='cuda')
        rand_all = torch.randperm(N_all, device='cuda') 
        
        # —————————————— # 前向传播
        LossSum = 0.0
        for layer,expert in zip(self.layers,expertset):
            tgt,LossUnit = layer(tgt,src,tgt_mask,tgt_pad_mask,src_pad_mask,expert
                                ,B,T,D,N,idx_all,N_all,cap
                                ,dtype_mid,local_rank_mid,rand_all)
            LossSum = LossSum + LossUnit
        
        # —————————————— # 输出
        tgt = self.EndLayerNorm(tgt)
        output = self.fc_out(tgt)
        return output,LossSum