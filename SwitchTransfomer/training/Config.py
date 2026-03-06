from torch.nn.parallel import DistributedDataParallel as DDP
from training.Encoder import *
from training.Decoder import *
from training.ExpertsSet import *
from training.Transfomer  import *
from training.SwitchTransfomer  import *

class Config():
    def __init__ (self,args,PadId,device,ProcessId,group):
        self.d_model=args.d_model
        self.h = args.HeadNum
        self.d_ff = args.d_ff
        self.dropout = args.dropout
        self.N = args.N
        self.vocab_size = args.vocab_size
        self.ExpertNum = args.ExpertNum
        self.GpuNum = args.GpuNum
        self.local_ExpertNum = args.ExpertNum // args.GpuNum 
        self.capacity_factor = args.capacity_factor
        self.sigma = args.sigma
        self.PadId = PadId
        self.device = device
        self.ProcessId = ProcessId
        self.group = group
    def make_model(self):
        # —————————————— # 实例化模型
        encoder = Encoder(self.d_model,self.h,self.N,self.vocab_size,self.ExpertNum,self.local_ExpertNum,self.GpuNum,self.dropout,self.ProcessId,self.capacity_factor,self.sigma,self.group)
        decoder = Decoder(self.d_model,self.h,self.N,self.vocab_size,self.ExpertNum,self.local_ExpertNum,self.GpuNum,self.dropout,self.ProcessId,self.capacity_factor,self.sigma,self.group)
        transfomer = Transfomer(encoder,decoder,self.PadId,self.h,self.device).to(self.device)
        expertsset = ExpertsSet(self.d_model,self.d_ff,self.local_ExpertNum,self.N).to(self.device)
        # —————————————— # 初始化参数
        transfomer.apply(init_weights)
        expertsset.apply(init_weights)
        # —————————————— # DDP包装 
        transfomer = DDP(transfomer)
        return SwitchTransfomer(transfomer,expertsset)