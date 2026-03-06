import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist 
from torch.nn import init

# —————————————— # 定义计时器
def CalculationTime(SumS):
    h = SumS // 3600
    m = (SumS - h*3600)//60
    s = int(SumS -h*3600 - m*60)
    return h,m,s
class Timer:
    def __init__(self):
        self.StartTime = None
    @property
    def start(self):
        self.StartTime = time.time()
    @property
    def end(self):
        h,m,s = CalculationTime(time.time()-self.StartTime)
        if h!=0:
            print(f'  ---------->  耗时 {h}h{m}m{s}s')
            return
        if m!=0:
            print(f'  ---------->  耗时 {m}m{s}s')
            return
        print(f'  ---------->  耗时 {s}s')

# —————————————— # 申请参数
def nn_LayerNorm(d_model):
    tamp = nn.LayerNorm(d_model)
    init.constant_(tamp.weight, 1.0)  # 权重设为1
    init.constant_(tamp.bias  , 0.0)  # 偏置设为0
    return tamp

# —————————————— # 初始化权重
def init_weights(model: nn.Module):
  if hasattr(model, 'weight') and model.weight.dim() > 1:
    nn.init.xavier_uniform_(model.weight.data)

# —————————————— # 获取进程信息
def GetProcessId():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl" , device_id = local_rank)
    ProcessId  = dist.get_rank()    # 当前进程的编号
    group =  dist.group.WORLD
    
    return ProcessId,group

# —————————————— # 学习率
class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000, factor: float = 1.0, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # PyTorch 的 _LRScheduler 内部用 last_epoch 记录 step 索引（从0开始）
        step = max(self.last_epoch + 1, 1)
        scale = (self.factor * (self.d_model ** -0.5) *
                 min(step ** -0.5, step * (self.warmup_steps ** -1.5)))
        return [base_lr * scale for base_lr in self.base_lrs]

def beam_decode(model, src_real,src_imag,src_pad_mask, BosId, EosId, PadId, max_len=300, beam_size=5, alpha=0.6):
    device = src_real.device

    beams = [(torch.tensor([BosId], device=device), 0.0, False)]
    finished = [] # 这里放结束了的句子

    for _ in range(max_len):
        candidates = [] # 这里放这个句子的候选

        for tokens, logp, is_fin in beams:
            if is_fin:
                candidates.append((tokens, logp, True))
                continue

            tgt = tokens.unsqueeze(0)  # (1, T)
            out = model.Transfomer.forward_Decoder(tgt,src_real,src_imag,src_pad_mask,model.ExpertsSet_real,model.ExpertsSet_imag)
            logits = out[0, -1, :]     # (V,)

            # 禁止 pad（可选但推荐）
            logits[PadId] = -1e9 # 把空格给禁掉

            lprobs = F.log_softmax(logits, dim=-1)  # (V,)
            topk_logp, topk_ids = torch.topk(lprobs, beam_size) # 获取lprobs中最大的beam_size个数据(要注意获取的id就已经是下一个token_id了)

            for k in range(beam_size):
                nid = topk_ids[k].item() # 获取id
                nlogp = logp + topk_logp[k].item() # 求分数和
                ntokens = torch.cat([tokens, torch.tensor([nid], device=device)]) #合并句子
                nfin = (nid == EosId)
                candidates.append((ntokens, nlogp, nfin))

        # 选出新的 beams（先按 length penalty 排序）
        def lp(length):
            # GNMT length penalty
            return ((5.0 + length) / 6.0) ** alpha

        candidates.sort(key=lambda x: x[1] / lp(len(x[0])), reverse=True)
        beams = candidates[:beam_size]

        # 收集 finished
        finished.extend([b for b in beams if b[2]])
        # 如果 beam 都 finished 了就提前停
        if all(b[2] for b in beams):
            break

    # 选最终输出
    pool = finished if len(finished) > 0 else beams

    def score(item):
        tokens, logp, _ = item
        return logp / (((5.0 + len(tokens)) / 6.0) ** alpha)

    best = max(pool, key=score)[0]

    # 去掉开头 BosId；去掉末尾 EosId（如果有）
    best_ids = best.tolist()
    if len(best_ids) > 0 and best_ids[0] == BosId:
        best_ids = best_ids[1:]
    if len(best_ids) > 0 and best_ids[-1] == EosId:
        best_ids = best_ids[:-1]

    return best_ids

def str2bool(v):
    if isinstance(v,bool) : return v
    if v.lower() in ("yes","true","t","1","y","on"):
        return True
    return False

def str2int(v):
    if isinstance(v,int) : return v
    return int(v)

def str2float(v):
    if isinstance(v,float) : return v
    return float(v)
        

