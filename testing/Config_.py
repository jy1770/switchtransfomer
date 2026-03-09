from testing.Encoder_ import *
from testing.Decoder_ import *
from testing.ExpertsSet_ import *
from testing.Transfomer_  import *
from testing.SwitchTransfomer_  import *

class Config():
    def __init__ (self,args,PadId,device):
        self.DataPath = args.DataPath
        self.num = args.num
        self.SrcName = args.SrcName
        self.TgtName = args.TgtName
        self.ExpertsFileNum = args.ExpertsFileNum

        self.d_model=args.d_model
        self.GpuNum = args.GpuNum
        self.h = args.HeadNum
        self.d_ff = args.d_ff
        self.N = args.N
        self.vocab_size = args.vocab_size
        self.ExpertNum = args.ExpertNum
        self.PadId = PadId
        self.device = device

    def load_model(self):
        local_ExpertNum = self.ExpertNum // self.GpuNum
        devices=[]
        for gpuidx in range(self.GpuNum):
            devices += [f'cuda:{gpuidx}']*local_ExpertNum
        # —————————————— # 实例化模型
        encoder = Encoder(self.d_model,self.h,self.N,self.vocab_size,self.ExpertNum)
        decoder = Decoder(self.d_model,self.h,self.N,self.vocab_size,self.ExpertNum)
        transfomer = Transfomer(encoder,decoder,self.PadId,self.h,self.device)
        expertsset = ExpertsSet(self.d_model,self.d_ff,self.ExpertNum,devices,self.N)

        # —————————————— # 导入共享参数
        checkpoint = torch.load(f'{self.DataPath}/.pt/switchtransfomer.Transfomer_{self.d_model}_{self.SrcName}_{self.TgtName}_{self.num}.pt', map_location="cpu")
        transfomer.load_state_dict(checkpoint, strict=False)
        
        # —————————————— # 导入专家参数
        idx = 0
        expertsset_state_dict = expertsset.state_dict()
        for i in range(self.ExpertsFileNum):
            # 导入专家参数
            checkpoint = torch.load(f'{self.DataPath}/.pt/switchtransfomer.ExpertsSet{i+1:>03}-of-{self.ExpertsFileNum:>03}_{self.d_model}_{self.SrcName}_{self.TgtName}_{self.num}.pt', map_location="cpu")
            for expert_idx in range(self.ExpertNum // self.ExpertsFileNum):
                for Name in ['src','tgt']:
                    for n in range(self.N):
                        for w in ['w_1','w_2']:
                            for suffix in ['weight','bias']:
                                key = f'{Name}.{n}.experts.{idx}.{w}.{suffix}'
                                value = f'{Name}.{n}.experts.{expert_idx}.{w}.{suffix}'
                                # print(f'{key:<30} <------- {value:<30}')
                                expertsset_state_dict[key] = checkpoint[value]
                idx += 1
        expertsset.load_state_dict(expertsset_state_dict, strict=False)

        # —————————————— # 导入gpu
        transfomer = transfomer.to('cuda:0')
        for n,Experts in enumerate(expertsset.src) :
            for expert_idx,expert in enumerate(Experts.experts):
                    expertsset.src[n].experts[expert_idx] = expert.to(devices[expert_idx])
        for n,Experts in enumerate(expertsset.tgt) :
            for expert_idx,expert in enumerate(Experts.experts):
                    expertsset.tgt[n].experts[expert_idx] = expert.to(devices[expert_idx])

        # —————————————— # 组合模型
        switchtransfomer = SwitchTransfomer(transfomer,expertsset)
        return switchtransfomer