import torch
import os 
import time
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
from training.DataSet import *
from training.Config  import *
from Function.Function import *
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.amp import autocast
from tqdm import tqdm

def split_decay(named_params):
    decay, no_decay = [], []
    for n, p in named_params:
        if (not p.requires_grad) or (p is None):
            continue
        if n.endswith(".bias") or ("LayerNorm" in n) or ("layer_norm" in n) or ("layernorm" in n):
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay

def make_param_groups(args,shared_named,router_named,expert_named):
    param_groups = []
    # —————————————— # shared
    shared_decay, shared_no_decay = split_decay(shared_named)
    if shared_decay:
        param_groups.append({"params": shared_decay, "lr": args.lr_shared, "weight_decay": args.wd_shared})
    if shared_no_decay:
        param_groups.append({"params": shared_no_decay, "lr": args.lr_shared, "weight_decay": 0.0})
    # —————————————— # router
    router_decay, router_no_decay = split_decay(router_named)
    if router_decay:
        param_groups.append({"params": router_decay, "lr": args.lr_router, "weight_decay": args.wd_router})
    if router_no_decay:
        param_groups.append({"params": router_no_decay, "lr": args.lr_router, "weight_decay": 0.0})
    # —————————————— # expert
    expert_decay, expert_no_decay = split_decay(expert_named)
    if expert_decay:
        param_groups.append({"params": expert_decay, "lr": args.lr_expert, "weight_decay": args.wd_expert})
    if expert_no_decay:
        param_groups.append({"params": expert_no_decay, "lr": args.lr_expert, "weight_decay": 0.0})
    return param_groups

def make_named(switchtransformer):
    switchtransformer_named = list(switchtransformer.transformer.named_parameters())
    shared_named = [(n, p) for (n, p) in switchtransformer_named if "router" not in n]
    router_named = [(n, p) for (n, p) in switchtransformer_named if "router" in n]
    expert_named = list(switchtransformer.expertset.named_parameters())
    return shared_named,router_named,expert_named

def save_model(args,switchtransformer,ProcessId,steps):
    torch.save(switchtransformer.expertset.state_dict(),f'{args.DataPath}/.pt/switchtransformer.expertset{ProcessId+1:>03}-of-{args.GpuNum:>03}_{args.d_model}_{args.SrcName}_{args.TgtName}_{str(steps)}.pt')
    if ProcessId == 0:
        torch.save(switchtransformer.transformer.module.state_dict(),f'{args.DataPath}/.pt/switchtransformer.transformer_{args.d_model}_{args.SrcName}_{args.TgtName}_{str(steps)}.pt')

def get_PadId(args):
    BPEPath = f'{args.DataPath}Data/TrainData/BPE-{args.SrcName}-{args.TgtName}.model'
    BPE = spm.SentencePieceProcessor(BPEPath) # 导入BPE
    return BPE.pad_id()

def function(args: argparse.Namespace):
    # —————————————— # 打开f32加速
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    
    # —————————————— # 初始化参数和文件
    ProcessId,group = GetProcessId()
    PadId = get_PadId(args)
    if ProcessId == 0:
        TrainLossPath = f'{args.DataPath}/TrainLoss/TrainLoss_1.txt'
        with open(TrainLossPath , 'w' ,encoding="utf-8") as f:
            f.write(f'LossSteps = {args.LossSteps}\n')
    shared_expert_LossSum,router_LossSum = 0.0,0.0 

    # —————————————— # 制作数据生成器
    modeldataset  = ModelDataSet(args)
    batch_sampler = ModelSampler(modeldataset, args, ProcessId)
    loader = DataLoader(modeldataset,batch_sampler=batch_sampler,collate_fn=partial(collate_fn, PadId=PadId),num_workers=3,pin_memory=True)

    # —————————————— # 制作模型
    modelconfig = Config(args,PadId,ProcessId,group)
    switchtransformer = modelconfig.make_model()

    # —————————————— # 配置参数优化器
    shared_named,router_named,expert_named = make_named(switchtransformer)
    param_groups = make_param_groups(args,shared_named,router_named,expert_named)
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=args.warmup_steps,num_training_steps=args.max_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=PadId, label_smoothing=0.1)

    # —————————————— # 预先缓存参数列表
    shared_params = [p for _, p in shared_named]
    expert_params = [p for _, p in expert_named]
    router_params = [p for _, p in router_named]

    # —————————————— # 开始训练
    switchtransformer.train()
    pbar = tqdm(total=args.max_steps, desc="Steps", disable = (ProcessId!=0), mininterval=1) ; 
    steps = 0
    #try:
    if True:
        while True:
            for batch in loader:
                # —————————————— # 初始化参数
                src,tgt = batch.src.cuda(),batch.tgt.cuda()
                optimizer.zero_grad(set_to_none=True)

                # —————————————— # 前向传播
                with autocast(device_type='cuda',dtype=torch.bfloat16):
                    out,router_loss = switchtransformer(src,tgt[:, :-1])
                    out = out.contiguous().view(-1, args.vocab_size)
                    tgt_y = tgt[:, 1:].contiguous().view(-1)
                    shared_expert_loss = criterion(out, tgt_y)
                del out,tgt_y

                # —————————————— # 计算梯度 & 梯度裁剪
                loss = shared_expert_loss + router_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_([p for p in shared_params if p.grad is not None], args.clip_shared)
                torch.nn.utils.clip_grad_norm_([p for p in expert_params if p.grad is not None], args.clip_expert)
                torch.nn.utils.clip_grad_norm_([p for p in router_params if p.grad is not None], args.clip_router)

                # —————————————— # 参数更新
                optimizer.step()
                scheduler.step()
                steps+=1
                pbar.update(1)
                
                # —————————————— # 统计损失值
                shared_expert_LossSum += shared_expert_loss.item()
                router_LossSum += router_loss.item()
                if steps%args.LossSteps==0:
                    mid_shared_expert_LossSum = torch.tensor(shared_expert_LossSum/args.LossSteps,device='cuda')
                    mid_router_LossSum = torch.tensor(router_LossSum/args.LossSteps,device='cuda')
                    dist.reduce(mid_shared_expert_LossSum, dst=0, op=dist.ReduceOp.AVG)
                    dist.reduce(mid_router_LossSum, dst=0, op=dist.ReduceOp.AVG)

                    if ProcessId == 0:
                        with open(TrainLossPath , 'a' ,encoding="utf-8") as f:
                            f.write(f'{mid_shared_expert_LossSum.item()}  {mid_router_LossSum.item()}\n')
                    shared_expert_LossSum,router_LossSum = 0.0,0.0
                del shared_expert_loss,router_loss,loss

                # —————————————— # 保存模型
                if steps%10000==0:
                    save_model(args,switchtransformer,ProcessId,steps)
                    torch.cuda.empty_cache() # 清理显存块
                    if steps >= args.max_steps: 
                        if ProcessId == 0:
                            with open(TrainLossPath , 'a' ,encoding="utf-8") as f:
                                f.write(str(pbar))
                                #pi=0
                        time.sleep(60)
                        os.system("su -c 'shutdown -h now'")
    #except:
    #    time.sleep(3)
    #    os.system("su -c 'shutdown -h now'") # 笑死，出现错误直接关机，省钱


def add_subparser(subparsers: argparse._SubParsersAction, parents=None):
    if parents is None:
        parents = []
    parser = subparsers.add_parser('train', help='数据处理',parents=parents)
    group = parser.add_argument_group('训练参数')
    # —————————————— # 训练参数
    group.add_argument("--LossSteps",default=1000, type=str2int  , help="记录loss的步数")
    group.add_argument("--S"      , default=18000, type=str2int  , help="单次训练token数")
    group.add_argument("--tf32"   , default=True , type=str2bool , help="是否打开tf32")
    group.add_argument("--lb_coef", default=0.1  , type=str2float, help="lb_loss系数")
    group.add_argument("--sort"   , default=True , type=str2bool , help="是否桶内排序")
    group.add_argument("--sigma"  , default=0.1  , type=str2float, help="")
    # —————————————— # 配置器参数
    group.add_argument("--max_steps"   , default=100000, type=str2int, help="")
    group.add_argument("--warmup_steps", default=8000  , type=str2int, help="")
    # —————— # 共享参数
    group.add_argument("--lr_shared"  , default=6e-4, type=str2float, help="")
    group.add_argument("--wd_shared"  , default=1e-2, type=str2float, help="")
    group.add_argument("--clip_shared", default=1.0 , type=str2float, help="")
    # —————— # 专家参数
    group.add_argument("--lr_expert"  , default=1.2e-3, type=str2float, help="")
    group.add_argument("--wd_expert"  , default=0.0   , type=str2float, help="")
    group.add_argument("--clip_expert", default=5.0   , type=str2float, help="")
    # —————— # 路由器
    group.add_argument("--lr_router"  , default=2e-4, type=str2float, help="")
    group.add_argument("--wd_router"  , default=0.0 , type=str2float, help="")
    group.add_argument("--clip_router", default=0.5 , type=str2float, help="")

    parser.set_defaults(func = function)