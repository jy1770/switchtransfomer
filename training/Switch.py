import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn.functional as distnn


class Switch(nn.Module):
    def __init__(self,d_model,ExpertNum,local_ExpertNum,GpuNum,ProcessId,capacity_factor,sigma,group):
        super().__init__()
        self.ExpertNum = ExpertNum
        self.local_ExpertNum = local_ExpertNum
        self.GpuNum = GpuNum
        self.ProcessId = ProcessId
        self.capacity_factor = capacity_factor
        self.sigma = sigma
        self.group = group

        self.Router = nn.Linear(d_model, self.ExpertNum, bias=False)

    def make_dispatch_mask(self,dispatch_mask,expert_idx,mask_flat,cap,gate):
        for e in range(self.ExpertNum):
            pos = (expert_idx == e) & mask_flat
            num = int(pos.sum().item())
            if num == 0:
                continue
            if num <= cap:
                dispatch_mask[pos] = True # 没有达到阈值，即直接设置为True
            else:
                idx = torch.nonzero(pos, as_tuple=False).squeeze(1) # 获取pos中True的下标
                g = gate.index_select(0, idx) # 根据下标获取g
                topk = torch.topk(g, k=cap, sorted=False).indices # 获取g中topk分数的下标
                dispatch_mask.index_fill_(0, idx[topk], True)
        return dispatch_mask


    def forward(self,x,experts,token_mask):
        # —————————————— # 参数初始化
        B, T, D = x.shape
        tokens_per_core = B * T  # 每个 GPU/core 上固定 token 数（包含 pad）
        x_flat_all = x.reshape(tokens_per_core, D) # [tokens_per_core,d_model] : 包含全部token
        flat_mask = token_mask.reshape(tokens_per_core)  # [tokens_per_core] : token_mask

        # —————————————— # 计算路由参数
        logits_clean = self.Router(x_flat_all)  # [tokens_per_core, ExpertNum]
        mult = 1.0 + (torch.rand_like(logits_clean) * 2.0 - 1.0) * self.sigma  # (1-σ,1+σ)
        if flat_mask is not None: # 对pad不进行干扰【我的测试的时候不用这个代码，用的是Switch_.py没有这个噪声】
            mult = torch.where(flat_mask.unsqueeze(-1), mult, torch.ones_like(mult))
        logits = logits_clean * mult

        # —————————————— # 计算probs,expert_idx,gate
        # —————— # 被噪声干扰过的，用于路由，增加泛化性
        probs = F.softmax(logits, dim=-1)  # 归一化
        expert_idx = probs.argmax(dim=-1)  # [tokens_per_core] : token 目标专家的下标idx
        gate = probs.gather(1, expert_idx[:, None]).squeeze(1)  # [tokens_per_core] : token 目标专家的分数gate
        # —————— # 没有被噪声干扰过，能真实的反应路由器self.Router的效果
        probs_clean = F.softmax(logits_clean, dim=-1)  # 归一化
        expert_idx_clean = probs_clean.argmax(dim=-1)  # [tokens_per_core] : token 目标专家的下标idx
        # gate_clean = probs_clean.gather(1, expert_idx_clean[:, None]).squeeze(1)  # [tokens_per_core] : token 目标专家的分数gate

        # —————————————— # 统计 N_all : 排除掉pad的全部token数
        idx_all = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
        N_all = int(idx_all.numel())

        # —————————————— # 计算cap : 真实的cap值
        cap = min(N_all , int((N_all*self.capacity_factor)/self.ExpertNum))

        # —————————————— # 计算 dispatch_mask 
        dispatch_mask = torch.zeros((tokens_per_core,), device=expert_idx.device, dtype=torch.bool)
        dispatch_mask = self.make_dispatch_mask(dispatch_mask,expert_idx,flat_mask,cap,gate)

        # —————————————— # 统计 N_send : 排除掉pad和被cap截断的token
        idx_send = torch.nonzero(dispatch_mask, as_tuple=False).squeeze(1)
        N_send = int(idx_send.numel())

        # —————————————— # 获取被筛选后的token数量 N(全局)
        num_all_local = torch.tensor([N_all], device='cuda', dtype=torch.int64)
        dist.all_reduce(num_all_local, op=dist.ReduceOp.SUM, group=self.group)
        num_all_global = int(num_all_local.item())
        '''
        num_kept_local = torch.tensor([N_send], device='cuda', dtype=torch.int64)
        dist.all_reduce(num_kept_local, op=dist.ReduceOp.SUM, group=self.group)
        num_kept_global = int(num_kept_local.item())
        '''
        
        # —————————————— # 获取被筛选后的数据 (all筛选出来的是用于计算loss的，使得负载均衡的,send筛选出来的是用于传入专家的)
        # x_all = x_flat_all.index_select(0, idx_all)      # [N_all, D]
        logits_all_clean = logits_clean.index_select(0, idx_all)     # [N_all, E]
        probs_all_clean = probs_clean.index_select(0, idx_all)       # [N_all, E]
        expert_all_clean = expert_idx_clean.index_select(0, idx_all) # [N_all]
        # gate_all = gate.index_select(0, idx_all)         # [N_all]

        x_send = x_flat_all.index_select(0, idx_send)      # [N_send, D]
        # logits_send = logits.index_select(0, idx_send)     # [N_send, E]
        # probs_send = probs.index_select(0, idx_send)       # [N_send, E]
        expert_send = expert_idx.index_select(0, idx_send) # [N_send]
        gate_send = gate.index_select(0, idx_send)         # [N_send]
        
        # —————————————— # lb_loss
        # —————— # 计算p
        p_sum = probs_all_clean.sum(dim=0)  # [E]
        p_sum = distnn.all_reduce(p_sum, op=dist.ReduceOp.SUM, group=self.group)
        # —————— # 计算f
        f_sum = F.one_hot(expert_all_clean, num_classes=self.ExpertNum).sum(dim=0).to(torch.int64)
        f_sum = distnn.all_reduce(f_sum, op=dist.ReduceOp.SUM, group=self.group)
        # —————— # lb_loss
        p,f = p_sum/num_all_global , f_sum/num_all_global
        lb_loss = self.ExpertNum * (p*f).sum()  # 标量

        # —————————————— # z_loss
        z_kept = torch.logsumexp(logits_all_clean, dim=-1)    # 只算 kept 的
        z_sum = (z_kept ** 2).sum()
        z_sum = distnn.all_reduce(z_sum, op=dist.ReduceOp.SUM, group=self.group)
        z_loss = z_sum / num_all_global
        '''
        # —————————————— # lb_loss
        # —————— # 计算p
        p_sum = probs_send.sum(dim=0)  # [E]
        p_sum = distnn.all_reduce(p_sum, op=dist.ReduceOp.SUM, group=self.group)
        # —————— # 计算f
        f_sum = F.one_hot(expert_send, num_classes=self.ExpertNum).sum(dim=0).to(torch.int64)
        f_sum = distnn.all_reduce(f_sum, op=dist.ReduceOp.SUM, group=self.group)
        # —————— # lb_loss
        p,f = p_sum/num_kept_global , f_sum/num_kept_global
        lb_loss = self.ExpertNum * (p*f).sum()  # 标量

        # —————————————— # z_loss
        z_kept = torch.logsumexp(logits_send, dim=-1)    # 只算 kept 的
        z_sum = (z_kept ** 2).sum()
        z_sum = distnn.all_reduce(z_sum, op=dist.ReduceOp.SUM, group=self.group)
        z_loss = z_sum / num_kept_global
        '''
        # —————————————— # 计算goal参数
        goal_eid  = expert_send %  self.local_ExpertNum # token目标进程中的专家idx
        goal_proc = expert_send // self.local_ExpertNum # token目标进程的idx

        # —————————————— # 按goal_proc排序,便于all_to_all切分
        perm = torch.argsort(goal_proc)
        x_perm = x_send.index_select(0, perm)
        gate_perm = gate_send.index_select(0, perm)
        goal_eid_perm = goal_eid.index_select(0, perm)
        goal_proc_perm = goal_proc.index_select(0, perm)

        # —————————————— # 制作接收统计(recv_counts)和发送统计(send_counts)
        send_counts = torch.bincount(goal_proc_perm, minlength=self.GpuNum).to(torch.int64)
        gathered = [torch.empty_like(send_counts) for _ in range(self.GpuNum)]
        dist.all_gather(gathered, send_counts, group=self.group)
        recv_counts = torch.stack(gathered, dim=0)[:, self.ProcessId].contiguous()

        send_splits = send_counts.cpu().tolist()
        recv_splits = recv_counts.cpu().tolist()
        total_recv = int(recv_counts.sum().item())

        # —————————————— # 第一次 all-to-all
        recv_x = torch.empty((total_recv, D), device='cuda', dtype=x.dtype)
        recv_gate = torch.empty((total_recv,), device='cuda', dtype=x.dtype)
        recv_local_eid = torch.empty((total_recv,), device='cuda', dtype=torch.int64)
        distnn.all_to_all_single(recv_x, x_perm,output_split_sizes=recv_splits,input_split_sizes=send_splits,group=self.group)
        distnn.all_to_all_single(recv_gate, gate_perm,output_split_sizes=recv_splits,input_split_sizes=send_splits,group=self.group)
        distnn.all_to_all_single(recv_local_eid, goal_eid_perm,output_split_sizes=recv_splits,input_split_sizes=send_splits,group=self.group)

        # —————————————— # 本地expert计算
        y_recv = torch.zeros((total_recv, D), device='cuda', dtype=x.dtype)
        y_recv = experts(y_recv, recv_x, recv_gate, recv_local_eid)

        # —————————————— # 第二次 all-to-all
        y_perm = torch.empty((N_send, D), device='cuda', dtype=x.dtype)
        distnn.all_to_all_single(y_perm, y_recv,output_split_sizes=send_splits,input_split_sizes=recv_splits,group=self.group)

        # —————————————— # 反排序回 idx_send 对应的顺序
        y_sel = torch.empty((N_send, D), device='cuda', dtype=x.dtype)
        y_sel.index_copy_(0, perm, y_perm)

        # —————————————— # 复原参数
        y_flat_all = torch.zeros((tokens_per_core, D), device='cuda', dtype=x.dtype)
        y_flat_all.index_copy_(0, idx_send, y_sel)
        y = y_flat_all.view(B, T, D)
        return y, lb_loss, z_loss
