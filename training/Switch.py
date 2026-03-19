import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn.functional as distnn
import random
'''
# 软lb_loss
def cal_lb_loss(probs,ExpertNum,eps): # 软lb_loss
    avg_prob = probs.mean(dim=0)  # [C]
    uniform = torch.full_like(avg_prob, 1.0 / ExpertNum)
    lb_loss = torch.sum(avg_prob * (torch.log(avg_prob + eps) - torch.log(uniform + eps)))
    return lb_loss
'''

# 硬lb_loss
def cal_lb_loss(probs_all,expert_idx_all,ExpertNum,N_all):
    p = probs_all.mean(dim=0)
    f = torch.bincount(expert_idx_all, minlength=ExpertNum)/N_all
    lb_loss = ExpertNum * (p*f).sum()
    return lb_loss 

@torch.no_grad()
def make_idx_send(expert_idx_all,cap,ExpertNum,local_rank_mid,rand_all):
    counts = torch.bincount(expert_idx_all, minlength=ExpertNum)  # 统计每个 expert 有多少 token
    if counts.max() <= cap:
        return local_rank_mid                # 如果所有 expert 的 token 数都 <= cap,那就不用截断了
    expert_idx_all_rand = expert_idx_all.index_select(0,rand_all) # 全局随机
    order = torch.argsort(expert_idx_all_rand,stable=True)        # 按 expert 排序
    starts = torch.cumsum(counts, dim=0) - counts                 # 计算每组起点
    local_rank = local_rank_mid - torch.repeat_interleave(starts,counts)
    keep = local_rank < cap                          # 每组保留前 cap 个
    idx_send = rand_all.index_select(0, order[keep]) # 写回

    return idx_send

class NetWorkRouter(nn.Module):
    def __init__(self,d_model,ExpertNum,sigma,lb_coef):
        super().__init__()
        # —————————————— # 模型参数
        self.linear  = nn.Linear(d_model,ExpertNum,bias=False)
        # —————————————— # 参数初始化
        self.ExpertNum = ExpertNum
        self.sigma = sigma
        self.lb_coef = lb_coef

    def forward(self,x_all,N_all):
        # —————————————— # 计算路由参数
        logits_all = self.linear(x_all)
        probs_all = torch.softmax(logits_all,dim=-1,dtype=logits_all.dtype)
        expert_idx_all = logits_all.argmax(dim=-1)

        # —————————————— # 计算损失函数
        lb_loss = cal_lb_loss(probs_all,expert_idx_all,self.ExpertNum,N_all)
        loss = self.lb_coef*lb_loss

        # —————————————— # 高斯扰动
        mult = 1.0 + (torch.rand_like(logits_all) * 2.0 - 1.0) * self.sigma  # (1-σ,1+σ)
        logits_all = logits_all * mult
        probs_all = torch.softmax(logits_all,dim=-1,dtype=logits_all.dtype)
        expert_idx_all = logits_all.argmax(dim=-1)

        del logits_all,lb_loss
        return probs_all,expert_idx_all,loss

class Switch(nn.Module):
    def __init__(self,d_model,ExpertNum,local_ExpertNum,GpuNum,ProcessId,sigma,group,lb_coef):
        super().__init__()
        self.ExpertNum = ExpertNum
        self.local_ExpertNum = local_ExpertNum
        self.GpuNum = GpuNum
        self.ProcessId = ProcessId
        self.group = group
        self.router = NetWorkRouter(d_model,ExpertNum,sigma,lb_coef)

        self.mid = torch.tensor([i for i in range(self.ExpertNum)],dtype=torch.int64,device='cuda')

    def forward(self,x,experts,B,T,D,N,idx_all,N_all,cap,dtype_mid,local_rank_mid,rand_all):
        # —————————————— # 参数初始化 & 路由计算
        x_falt = x.reshape(N, D)
        x_all = x_falt.index_select(0,idx_all)
        probs_all,expert_idx_all,loss = self.router(x_all,N_all)
        
        # —————————————— # 制作 idx_send & N_send
        idx_send = make_idx_send(expert_idx_all,cap,self.ExpertNum,local_rank_mid,rand_all)
        N_send = int(idx_send.numel())

        # —————————————— # 获取筛选的值 & 计算目标专家id和进程
        x_send = x_all.index_select(0, idx_send)
        probs_send = probs_all.index_select(0, idx_send)
        expert_idx_send = expert_idx_all.index_select(0, idx_send)
        gate_send = probs_send.gather(1, expert_idx_send[:, None]).squeeze(1)

        rand_pos = torch.tensor(random.sample(range(N_send),self.ExpertNum),dtype=torch.long,device='cuda')
        expert_idx_send[rand_pos] = self.mid

        goal_expert_idx  = expert_idx_send % self.local_ExpertNum
        goal_proc = expert_idx_send // self.local_ExpertNum

        # —————————————— # 制作接收统计(recv_counts)和发送统计(send_counts)
        send_counts = torch.bincount(goal_proc, minlength=self.GpuNum).to(torch.int64)
        gathered = [torch.empty_like(send_counts) for _ in range(self.GpuNum)]
        dist.all_gather(gathered, send_counts, group=self.group)
        recv_counts = torch.stack(gathered, dim=0)[:, self.ProcessId].contiguous()
        send_splits = send_counts.cpu().tolist()
        recv_splits = recv_counts.cpu().tolist()
        total_recv = int(recv_counts.sum().item())

        # —————————————— # 第一次 all-to-all
        recv_x = torch.empty((total_recv, D), device='cuda', dtype=x.dtype)
        recv_x = distnn.all_to_all_single(recv_x, x_send,output_split_sizes=recv_splits,input_split_sizes=send_splits,group=self.group)

        recv_gate = torch.empty((total_recv,), device='cuda', dtype=dtype_mid)
        recv_gate = distnn.all_to_all_single(recv_gate, gate_send,output_split_sizes=recv_splits,input_split_sizes=send_splits,group=self.group)
        
        recv_local_eid = torch.empty((total_recv,), device='cuda', dtype=torch.int64)
        recv_local_eid = distnn.all_to_all_single(recv_local_eid, goal_expert_idx,output_split_sizes=recv_splits,input_split_sizes=send_splits,group=self.group)
        del x_send,gate_send,goal_expert_idx

        # —————————————— # 本地expert计算
        y_recv = torch.zeros((total_recv,D),device='cuda',dtype=dtype_mid)
        y_recv = experts(y_recv,recv_x,recv_gate,recv_local_eid)

        # —————————————— # 第二次 all-to-all
        y_perm = torch.empty((N_send, D), device='cuda', dtype=dtype_mid)
        y_perm = distnn.all_to_all_single(y_perm, y_recv,output_split_sizes=send_splits,input_split_sizes=recv_splits,group=self.group)

        # —————————————— # 写回
        final_idx = idx_all.index_select(0, idx_send)
        y_flat_all = torch.zeros((N, D), device='cuda', dtype=dtype_mid)#x.dtype)
        y_flat_all.index_copy_(0,final_idx,y_perm)
        y = y_flat_all.view(B, T, D)
        return y,loss
