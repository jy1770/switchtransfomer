import struct
import torch
import torch.distributed as dist
import numpy as np
import mmap
import sentencepiece as spm
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from tqdm import tqdm

class IndexedBinReader:
    def __init__(self,bin_path,idx_path,BosId,EosId):
        MAGIC = b"IDXBIN01"
        with open(idx_path, "rb") as f:
            magic = f.read(8)
            if magic != MAGIC:
                raise ValueError(f"bad magic: {magic}")
            self.SenNum = int(struct.unpack("<Q", f.read(8))[0])
            self.offsets = np.frombuffer(f.read(self.SenNum * 8), dtype=np.uint64)
            self.lens    = np.frombuffer(f.read(self.SenNum * 4), dtype=np.uint32)
        self.bin_f = open(bin_path, "rb")
        self.mm  = mmap.mmap(self.bin_f.fileno(), 0, access=mmap.ACCESS_READ)
        self.arr = np.frombuffer(self.mm, dtype=np.int32)  # 零拷贝视图
        # —————————————— # 保存参数
        self.BosId = np.array([BosId], dtype=np.int32)
        self.EosId = np.array([EosId], dtype=np.int32)
    def __len__(self):
        return self.SenNum
    def get(self, i):
        off = int(self.offsets[i])
        ln  = int(self.lens[i])
        view = self.arr[off: off + ln]
        return np.concatenate([self.BosId,view,self.EosId])
    def close(self):
        self.mm.close()
        self.bin_f.close()
    def __del__(self):
        try: 
            self.close()
        except: 
            pass

class ModelDataSet(Dataset):
    def __init__(self,args):
        # —————————————— # 地址参数初始化
        src_bin = f'{args.DataPath}Data/TrainData/{args.SrcName}-{args.TgtName}.src.bin'
        tgt_bin = f'{args.DataPath}Data/TrainData/{args.SrcName}-{args.TgtName}.tgt.bin'
        src_idx = f'{args.DataPath}Data/TrainData/{args.SrcName}-{args.TgtName}.src.idx'
        tgt_idx = f'{args.DataPath}Data/TrainData/{args.SrcName}-{args.TgtName}.tgt.idx'
        BPEPath = f'{args.DataPath}Data/TrainData/BPE-{args.SrcName}-{args.TgtName}.model'
        # —————————————— # 导入BPE
        BPE = spm.SentencePieceProcessor(BPEPath) # 导入BPE
        BosId,EosId = BPE.bos_id(),BPE.eos_id()
        # —————————————— # 制作类
        self.src = IndexedBinReader(src_bin,src_idx,BosId,EosId)
        self.tgt = IndexedBinReader(tgt_bin,tgt_idx,BosId,EosId)
    def __len__(self):
        return self.src.SenNum
    def __getitem__(self, idx):
        return self.src.get(idx),self.tgt.get(idx)

class ModelSampler(Sampler):
    def __init__(self,modeldataset : ModelDataSet,args, ProcessId):
        self.src_lens = modeldataset.src.lens
        self.tgt_lens = modeldataset.tgt.lens
        self.SenNum = modeldataset.src.SenNum
        self.S = args.S
        self.GpuNum = args.GpuNum
        self.sort = args.sort
        self.ProcessId = ProcessId
    def __iter__(self):
        # —————————————— # 全局打乱
        if self.ProcessId == 0:
            idx = torch.from_numpy(np.random.permutation(self.SenNum)).to(torch.int64)
        else:
            idx = torch.empty(self.SenNum, dtype=torch.int64)
        idx = idx.to(f'cuda:{self.ProcessId}')
        dist.broadcast(idx, src=0)
        IdxArr = idx.cpu().numpy()
        del idx
        # —————————————— # 取出当前进程的参数
        LocalIdxArr = IdxArr[self.ProcessId::self.GpuNum]
        LocalSrcLens = self.src_lens[LocalIdxArr]
        LocalTgtLens = self.tgt_lens[LocalIdxArr]
        LocalSenLens = LocalSrcLens + LocalTgtLens
        # —————————————— # 桶内排序
        if self.sort:
            idx = np.lexsort((LocalSrcLens, LocalSenLens))
            LocalIdxArr = LocalIdxArr[idx]
            LocalSrcLens = LocalSrcLens[idx]
            LocalTgtLens = LocalTgtLens[idx]
            LocalSenLens = LocalSenLens[idx]
        # —————————————— # 制作SamplerList
        SamplerList,SamplerUnit = [],[]
        MaxSrcLen,MaxTgtLen,BatchSize = 0,0,0
        for i in range(len(LocalIdxArr)):
            MaxSrcLen,MaxTgtLen = max(MaxSrcLen,LocalSrcLens[i]),max(MaxTgtLen,LocalTgtLens[i])
            BatchSize += 1
            if (MaxSrcLen+MaxTgtLen)*BatchSize >= self.S:
                SamplerList.append(SamplerUnit)
                SamplerUnit = [LocalIdxArr[i]]
                MaxSrcLen,MaxTgtLen = LocalSrcLens[i],LocalTgtLens[i]
                BatchSize = 1
            else:
                SamplerUnit.append(LocalIdxArr[i])
        else:
            SamplerList.append(SamplerUnit)
        # —————————————— # 获取不同的进程的数据长度
        LocalBatchNum = torch.tensor(len(SamplerList),device = f'cuda:{self.ProcessId}')
        dist.all_reduce(LocalBatchNum, op=dist.ReduceOp.MIN)
        self.LocalBatchNum = LocalBatchNum.cpu().item()
        del LocalBatchNum
        # —————————————— # 迭代输出
        for i in np.random.permutation(self.LocalBatchNum):
            yield  SamplerList[i]
    def __len__(self):
        return self.LocalBatchNum
    

class Batch():
    def __init__(self,BatchSize,MaxSrcLen,MaxTgtLen,PadId):
        self.src = torch.full((BatchSize, MaxSrcLen), PadId, dtype=torch.long)
        self.tgt = torch.full((BatchSize, MaxTgtLen), PadId, dtype=torch.long)

def collate_fn(samples,PadId):
    MaxSrcLen,MaxTgtLen = 0,0
    src,tgt = [],[]
    BatchSize = len(samples)
    for unit in samples:
        src.append(torch.tensor(unit[0]))
        tgt.append(torch.tensor(unit[1]))
        MaxSrcLen = max(MaxSrcLen,len(unit[0]))
        MaxTgtLen = max(MaxTgtLen,len(unit[1]))
    batch = Batch(BatchSize,MaxSrcLen,MaxTgtLen,PadId)
    for senidx in range(BatchSize):
         batch.src[senidx,:src[senidx].numel()] = src[senidx]
         batch.tgt[senidx,:tgt[senidx].numel()] = tgt[senidx]

    return batch