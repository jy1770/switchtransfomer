import torch.nn as nn
import torch
class FFNExpert(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__()
        self.w_1  = nn.Linear(d_model, d_ff)     # 第1层: d_model -> d_ff
        self.ReLU = nn.ReLU()                   # 激活函数(也可用 ReLU/Swish)
        self.w_2  = nn.Linear(d_ff, d_model)     # 第2层: d_ff -> d_model
    def forward(self, x):
        return self.w_2(self.ReLU(self.w_1(x)))

class Experts(nn.Module):
    def __init__(self,d_model,d_ff,ExpertsNum,devices):
        super().__init__()
        self.ExpertNum = ExpertsNum
        self.devices = devices
        self.experts = nn.ModuleList([FFNExpert(d_model,d_ff) for _ in range(ExpertsNum)])
    def forward(self,y_recv,x_flat,gate,local_eid):
        for le in range(self.ExpertNum):
            if y_recv.device != self.devices[le]: 
                y_recv = y_recv.to(self.devices[le])
                x_flat = x_flat.to(self.devices[le])
                gate = gate.to(self.devices[le])
                local_eid = local_eid.to(self.devices[le])

            pos = (local_eid == le).nonzero(as_tuple=False).squeeze(1)
            if pos.numel() == 0:
                continue
            g = gate[pos] # g: 这些 token 的 gate 权重
            y = self.experts[le](x_flat[pos])
            y = y * g.unsqueeze(-1)
            y_recv[pos] = y # 写回到对应位置；未写入的位置保持 0(等价于 overflow token 被丢弃)
        return y_recv.to('cuda:0')
    
class ExpertsSet(nn.Module):
    def __init__(self,d_model,d_ff,ExpertNum,devices,N):
        super().__init__()
        self.src = nn.ModuleList([Experts(d_model,d_ff,ExpertNum,devices) for _ in range(N)])
        self.tgt = nn.ModuleList([Experts(d_model,d_ff,ExpertNum,devices) for _ in range(N)])