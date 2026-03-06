import torch
import torch.nn as nn


class FFNExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.relu(self.w_1(x)))


class Experts(nn.Module):
    def __init__(self, d_model: int, d_ff: int, local_ExpertNum: int):
        super().__init__()
        self.local_ExpertNum = int(local_ExpertNum)
        self.experts = nn.ModuleList([FFNExpert(d_model, d_ff) for _ in range(self.local_ExpertNum)])

    def forward(self,y_recv,recv_x_flat,recv_gate,recv_local_eid):
        for le in range(self.local_ExpertNum):
            pos = (recv_local_eid == le).nonzero(as_tuple=False).squeeze(1)
            if pos.numel() == 0:
                continue
            g = recv_gate.index_select(0, pos)  # [n_i]
            y = self.experts[le](recv_x_flat.index_select(0, pos))  # [n_i, D]
            y = y * g.unsqueeze(-1)
            y_recv.index_copy_(0, pos, y)
        return y_recv

class ExpertsSet(nn.Module):
    def __init__(self, d_model: int, d_ff: int, local_ExpertNum: int, N: int):
        super().__init__()
        self.src = nn.ModuleList([Experts(d_model, d_ff, local_ExpertNum) for _ in range(N)])
        self.tgt = nn.ModuleList([Experts(d_model, d_ff, local_ExpertNum) for _ in range(N)])
