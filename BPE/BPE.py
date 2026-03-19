import argparse
import sentencepiece as spm
from tqdm import tqdm
import struct
import numpy as np
from pathlib import Path


def train(args: argparse.Namespace):
    # —————————————— # 地址初始化
    TrainBPE_SrcTgtPath = f'{args.DataPath}Data/TrainData/TrainBPE-{args.SrcName}-{args.TgtName}.txt'
    BPEPath  = f'{args.DataPath}Data/TrainData/BPE-{args.SrcName}-{args.TgtName}'

    # —————————————— # 训练BPE
    print(f'正在使用{TrainBPE_SrcTgtPath}训练BPE并保存在{BPEPath}')
    spm.SentencePieceTrainer.train(input = TrainBPE_SrcTgtPath, 
                                   model_prefix = BPEPath,
                                   vocab_size=args.vocab_size, 
                                   character_coverage=1.,
                                   unk_id=0,                  # 未知
                                   bos_id=1,                  # 句子开始
                                   eos_id=2,                  # 句子结束
                                   pad_id=3)                  # 空格

def write_ids(bin_f, ids):
    arr = np.asarray(ids, dtype=np.int32)
    bin_f.write(arr.tobytes(order="C"))

def write_idx(f, offsets, lens):
    n = len(offsets)
    MAGIC = b"IDXBIN01"
    f.write(MAGIC)
    f.write(struct.pack("<Q", n))
    f.write(np.asarray(offsets, dtype=np.uint64).tobytes())
    f.write(np.asarray(lens, dtype=np.uint32).tobytes())

def use(args: argparse.Namespace):
    # —————————————— # 地址初始化
    RaWSrcTgtPath = Path(f'{args.DataPath}Data/TrainData/Raw-{args.SrcName}-{args.TgtName}.txt')
    BPEPath  = f'{args.DataPath}Data/TrainData/BPE-{args.SrcName}-{args.TgtName}.model'
    # —————————————— # 打开文件
    src_bin = open(f'{args.DataPath}Data/TrainData/{args.SrcName}-{args.TgtName}.src.bin', "wb")
    tgt_bin = open(f'{args.DataPath}Data/TrainData/{args.SrcName}-{args.TgtName}.tgt.bin', "wb")
    src_idx = open(f'{args.DataPath}Data/TrainData/{args.SrcName}-{args.TgtName}.src.idx', "wb")
    tgt_idx = open(f'{args.DataPath}Data/TrainData/{args.SrcName}-{args.TgtName}.tgt.idx', "wb")
    # —————————————— # 参数初始化
    BPE = spm.SentencePieceProcessor(BPEPath)
    src_offsets,src_lens = [],[]
    tgt_offsets,tgt_lens = [],[]
    src_cursor,tgt_cursor = 0,0

    # —————————————— # 写入bin
    with RaWSrcTgtPath.open("r", encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="正在写入.bin"):
            try:
                s, t = line.rstrip("\n").split("\t")
            except:
                continue
            # —————— # 小写并编码
            s_ids ,t_ids= BPE.Encode(s.lower()),BPE.Encode(t.lower())
            # —————— # src写入
            src_offsets.append(src_cursor)
            src_lens.append(len(s_ids))
            write_ids(src_bin, s_ids)
            src_cursor += len(s_ids)
            # —————— # tgt写入
            tgt_offsets.append(tgt_cursor)
            tgt_lens.append(len(t_ids))
            write_ids(tgt_bin, t_ids)
            tgt_cursor += len(t_ids)
    # —————————————— # 写入idx
    write_idx(src_idx, src_offsets, src_lens)
    write_idx(tgt_idx, tgt_offsets, tgt_lens)
    # —————————————— # 关闭文件
    src_bin.close();tgt_bin.close();src_idx.close();tgt_idx.close()