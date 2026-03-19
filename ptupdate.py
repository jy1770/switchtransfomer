import torch
from collections import OrderedDict
from training.Encoder import *
from training.Decoder import *
from training.ExpertSet import *
from training.Transformer  import *
from training.SwitchTransformer  import *

class Config():
    def __init__(self):
        self.d_model = 512
        self.h = 8
        self.d_ff = 2048
        self.dropout = 0.1
        self.N = 6
        self.vocab_size = 32000
        self.ExpertNum = 8
        self.GpuNum = 8
        self.local_ExpertNum = 1
        self.capacity_factor = None
        self.sigma = None
        self.lb_coef = None
        self.PadId = None
        self.ProcessId = None
        self.group = None

    def make_model(self):
        # —————————————— # 实例化模型
        encoder = Encoder(self.d_model,self.h,self.N,self.vocab_size,self.ExpertNum,self.local_ExpertNum,self.GpuNum,self.dropout,self.ProcessId,self.capacity_factor,self.sigma,self.group,self.lb_coef)
        decoder = Decoder(self.d_model,self.h,self.N,self.vocab_size,self.ExpertNum,self.local_ExpertNum,self.GpuNum,self.dropout,self.ProcessId,self.capacity_factor,self.sigma,self.group,self.lb_coef)
        transformer = Transformer(encoder, decoder, self.PadId, self.h, self.N)
        expertset = ExpertsSet(self.d_model, self.d_ff, self.local_ExpertNum, self.N)
        return transformer,expertset
# —————————————— # 制作空模型
config = Config()
transformer,expertset = config.make_model()

# —————————————— # 地址初始化
SrcDataPath = 'C:/Users/Lenovo/Desktop/bs/Data/'
TgtDataPath = 'C:/Users/Lenovo/Desktop/jy/Data/'



if True:
    num = 10000


for i in range(10):
    num = (i+1)*10000
    

    # —————————————— # transformer更新
    srctransformer = torch.load(f'{SrcDataPath}.pt3/switchtransfomer.Transfomer_512_en_fr_{num}.pt', map_location="cpu")
    tgttransformer = OrderedDict()

    for key,value in srctransformer.items():
        if 'Encoder' in key:
            key = key.replace('Encoder', 'encoder')
        if 'Decoder' in key:
            key = key.replace('Decoder', 'decoder')
        if 'Router' in key:
            key = key.replace('Router', 'router.linear')
        tgttransformer[key] = value


    print('开始检查')
    for SrcName,TgtName in zip(tgttransformer,transformer.state_dict()):
        if SrcName != TgtName:
            print(f'{SrcName:<40}  {TgtName}')
    print('检查完毕')

    transformer.load_state_dict(tgttransformer, strict=True)
    torch.save(transformer.state_dict(),f'{TgtDataPath}.pt1/switchtransformer.transformer_512_en_fr_{num}.pt')



'''
GpuIdx = 1

srcexpertset = torch.load(f'{SrcDataPath}/.pt3/switchtransformer.ExpertsSet00{GpuIdx}-of-008_512_en_fr_{num}.pt', map_location="cpu")
tgtexpertset = OrderedDict()

#for name in srcexpertset:
#    print(name)

for key,value in srcexpertset.items():
    #if 'Encoder' in key:
    #    key = key.replace('Encoder', 'encoder')
    #if 'Decoder' in key:
    #    key = key.replace('Decoder', 'decoder')
    #if 'Router' in key:
    #    key = key.replace('Router', 'router.linear')
    tgtexpertset[key] = value

print('开始检查')
for SrcName,TgtName in zip(tgtexpertset,expertset.state_dict()):
    if SrcName != TgtName:
        print(f'{SrcName:<40}  {TgtName}')
print('检查完毕')
'''



'''

TgtDataPath = 'C:/Users/Lenovo/Desktop/jy/Data/'


for NumIdx in range(10):
    for GpuIdx in range(8):
        os.rename(f"{TgtDataPath}.pt1/switchtransfomer.ExpertsSet00{GpuIdx+1}-of-008_512_en_fr_{(NumIdx+1)*10000}.pt", f"{TgtDataPath}.pt1/switchtransformer.expertset00{GpuIdx+1}-of-008_512_en_fr_{(NumIdx+1)*10000}.pt")



'''