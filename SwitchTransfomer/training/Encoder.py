import torch
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
    def forward(self,src,src_pad_mask,experts):
        # —————————————— # 自注意力
        src_ = self.MhaLayerNorm(src)
        src_ = self.SelfMha(src_,src_,src_,key_padding_mask=src_pad_mask,need_weights=False)[0]
        src  = src + self.dropout(src_)
        # —————————————— # 全连接
        token_mask = ~src_pad_mask
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
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(f'cuda:{ProcessId}')
    def forward(self,src,src_pad_mask,expertsset):
        # —————————————— # 嵌入
        src = self.positionalencoding(self.tok_embedding(src)*self.scale)
        lb_loss_list ,z_loss_list=[],[]
        # —————————————— # 前向传播
        for i,layer in enumerate(self.layers):
            src,lb_loss,z_loss = layer(src,src_pad_mask,expertsset[i])
            lb_loss_list.append(lb_loss) ; z_loss_list.append(z_loss)
        # —————————————— # 输出
        src = self.EndLayerNorm(src)
        return src,torch.stack(lb_loss_list).mean(),torch.stack(z_loss_list).mean()





