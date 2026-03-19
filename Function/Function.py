import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist 
from torch.nn import init

# —————————————— # LayerNorm
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
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",device_id=local_rank)
    ProcessId  = dist.get_rank()     # 当前进程的编号
    group =  dist.group.WORLD
    return ProcessId,group

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