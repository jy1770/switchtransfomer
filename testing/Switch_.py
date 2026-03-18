import torch
import torch.nn as nn
import torch.nn.functional as F

class Switch(nn.Module):
    def __init__(self,d_model,ExpertNum):
        super().__init__()
        self.ExpertNum = ExpertNum
        self.device='cuda:0'
        self.Router  = nn.Linear(d_model, ExpertNum, bias=False)

    def forward(self,x,experts,token_mask):
        # —————————————— # 参数初始化
        B, T, D = x.shape
        x_flat_all = x.reshape(B * T, D)
        mask_flat = token_mask.reshape(B * T)

        # —————————————— # 获取非PadId的idx,和干净的x_flat
        idx_flat = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
        N = int(idx_flat.numel())
        x_flat = x_flat_all.index_select(0, idx_flat)  # (N_eff, D)

        # —————————————— # 获取最高分数及其下标
        logits = self.Router(x_flat)           # (N, D)*(D,E) = (N, E)
        probs = F.softmax(logits, dim=-1)      # (N, E)
        expert_idx = probs.argmax(dim=-1)      # 获取分数对应的下标
        gate = probs.gather(1, expert_idx[:, None]).squeeze(1)  # 获取分数

        # —————————————— # 本地专家计算
        y = torch.zeros((N, D), device='cuda:0', dtype=x.dtype)
        y = experts(y,x_flat,gate,expert_idx)
        
        # —————————————— # 返回完整参数
        y_flat_all = torch.zeros((B * T, D), device=self.device, dtype=x.dtype)
        y_flat_all.index_copy_(0, idx_flat, y)
        y = y_flat_all.view(B, T, D)
        return y
