import re
import os
import torch # PyTorch 版本: 2.9.1+cu128
import argparse
import sacrebleu
import sentencepiece as spm
from tqdm import tqdm
from testing.Config_ import *

SEG_RE = re.compile(r"<seg[^>]*>(.*?)</seg>")

def read_sgm_segs(path: str, lowercase: bool = False):
    segs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = SEG_RE.search(line)
            if m:
                txt = m.group(1).strip()
                if lowercase:
                    txt = txt.lower()
                segs.append(txt)
    return segs

def get_FilePath(args):
    SrcFilePath = f'{args.DataPath}Data/TestData/newstest2014-{args.SrcName+args.TgtName}-src.{args.SrcName}.sgm'
    RefFilePath = f'{args.DataPath}Data/TestData/newstest2014-{args.SrcName+args.TgtName}-ref.{args.TgtName}.sgm'
    if not os.path.exists(SrcFilePath):
        SrcFilePath = f'{args.DataPath}Data/TestData/newstest2014-{args.TgtName+args.SrcName}-src.{args.SrcName}.sgm'
        RefFilePath = f'{args.DataPath}Data/TestData/newstest2014-{args.TgtName+args.SrcName}-ref.{args.TgtName}.sgm'

    return SrcFilePath,RefFilePath

def Greedy(args: argparse.Namespace):
    # —————————————— # 初始化地址参数
    BPEPath  = f'{args.DataPath}Data/TrainData/BPE-{args.SrcName}-{args.TgtName}.model'
    SrcFilePath,RefFilePath = get_FilePath(args)

    # —————————————— # 初始化参数
    device, tgt_texts = 'cuda:0', []
    BPE = spm.SentencePieceProcessor(BPEPath)
    BosId, EosId, PadId = BPE.bos_id(), BPE.eos_id(), BPE.pad_id()

    switchtransfomerconfig = Config(args, PadId, device)
    switchtransfomer = switchtransfomerconfig.load_model()

    # —————————————— # TF32
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # —————————————— # 获取参考集
    src_lines = read_sgm_segs(SrcFilePath, lowercase=True)
    ref_texts = read_sgm_segs(RefFilePath, lowercase=True)
    print(f'SrcFilePath : {SrcFilePath}')
    print(f'RefFilePath : {RefFilePath}')

    # —————————————— # 预先 BPE 编码（避免循环里频繁 Encode）
    src_ids_list = []
    for s in src_lines:
        ids = [BosId] + BPE.Encode(s.rstrip().lower())  + [EosId]
        src_ids_list.append(ids)

    switchtransfomer = switchtransfomer.eval()

    with torch.inference_mode(): # inference_mode 比 no_grad 更快/更省
        for st in tqdm(range(0, len(src_ids_list), args.batch_size),desc=f'正在评估:transfomer_{args.d_model}_{args.SrcName}_{args.TgtName}_{args.num}.pt'):
            batch_ids = src_ids_list[st: st + args.batch_size]
            B = len(batch_ids)
            max_src = max(len(x) for x in batch_ids)

            # ——— src padding -> (B, Lsrc)
            src = torch.full((B, max_src), PadId, device=device, dtype=torch.long)
            for i, ids in enumerate(batch_ids):
                src[i, :len(ids)] = torch.tensor(ids, device=device)

            # ——— Encoder 一次跑完
            src_enc, src_pad_mask = switchtransfomer.Transfomer.forward_Encoder(src, switchtransfomer.ExpertsSet)

            # ——— Greedy 解码
            tgt = torch.full((B, 1), BosId, device=device, dtype=torch.long)
            finished = torch.zeros((B,), device=device, dtype=torch.bool)
            
            for _ in range(args.max_len):
                out = switchtransfomer.Transfomer.forward_Decoder(tgt, src_enc, src_pad_mask, switchtransfomer.ExpertsSet)  # (B, t, V)

                # 取最后一步 logits
                logits = out[:, -1, :].clone()  # (B, V)

                # ✅关键1：禁止生成 PAD（建议也禁 BOS）
                logits[:, PadId] = -1e9
                logits[:, BosId] = -1e9  # 可选但强烈推荐

                # greedy 选 token
                next_id = logits.argmax(dim=-1)  # (B,)

                # 已完成的句子不再增长：对“之前就 finished 的样本”强制输出 Pad（保持张量形状稳定）
                next_id = torch.where(finished, torch.full_like(next_id, PadId), next_id)

                # ✅关键2：finished 只由 EOS 决定（不要把 PAD 当 EOS）
                newly_finished = (next_id == EosId)
                finished = finished | newly_finished

                # append
                tgt = torch.cat([tgt, next_id.unsqueeze(1)], dim=1)

                if finished.all():
                    break
            # ——— 转回文本：去掉 Bos；遇到 Eos/Pad 截断
            tgt_cpu = tgt[:, 1:].tolist()
            for seq in tgt_cpu:
                cut = []
                for tid in seq:
                    if tid == EosId or tid == PadId:
                        break
                    cut.append(tid)
                tgt_texts.append(BPE.DecodeIds(cut))

    bleu = sacrebleu.corpus_bleu(tgt_texts, [ref_texts])
    print(bleu)
    print("BLEU =", bleu.score)

# pred_tgt = beam_decode(transfomer, src,src_pad_mask , BosId, EosId, PadId, max_len=args.max_len, beam_size=args.beam_size, alpha=args.alpha)
def Beam(args: argparse.Namespace):
    # —————————————— # 初始化地址参数
    BPEPath  = f'{args.DataPath}Data/TrainData/BPE-{args.SrcName}-{args.TgtName}.model'
    SrcFilePath,RefFilePath = get_FilePath(args)

    # —————————————— # 初始化参数
    device,tgt_texts = 'cuda:0',[]
    BPE = spm.SentencePieceProcessor(BPEPath)
    BosId,EosId,PadId = BPE.bos_id(),BPE.eos_id(),BPE.pad_id()
    switchtransfomerconfig = Config(args,PadId,device)
    switchtransfomer = switchtransfomerconfig.load_model()

    # —————————————— # 获取参考集
    src_lines = read_sgm_segs(SrcFilePath, lowercase=True)   # 你想要全小写就 True
    ref_texts = read_sgm_segs(RefFilePath, lowercase=True)
    print(f'SrcFilePath : {SrcFilePath}')
    print(f'RefFilePath : {RefFilePath}')

    # —————————————— # 开始评估
    switchtransfomer = switchtransfomer.eval()
    with torch.no_grad():
        for src in tqdm(src_lines, total=len(ref_texts),desc=f'正在评估:transfomer_{args.d_model}_{args.SrcName}_{args.TgtName}_{args.num}.pt'):
            # —————— # 初始化参数
            src = torch.tensor([[BosId] + BPE.Encode(src.rstrip().lower())], device=device)
            pred_tgt = []
            src,src_pad_mask = switchtransfomer.Transfomer.forward_Encoder(src,switchtransfomer.ExpertsSet)

            pred_tgt = beam_decode(switchtransfomer, src,src_pad_mask , BosId, EosId, PadId, max_len=args.max_len, beam_size=args.beam_size, alpha=args.alpha)
            # —————— # 处理预测tgt
            tgt_texts.append(BPE.DecodeIds(pred_tgt))
    
    # —————————————— # 打印BLEU
    bleu = sacrebleu.corpus_bleu(tgt_texts, [ref_texts])
    print(bleu)
    print("BLEU =", bleu.score)


def function(args: argparse.Namespace):
    if args.Greedy:
        print('Greedy评估')
        Greedy(args)
    if args.Beam:
        print('Beam评估')
        Beam(args)

def add_subparser(subparsers: argparse._SubParsersAction, parents=None):
    if parents is None:
        parents = []
    parser = subparsers.add_parser('test', help='数据处理',parents=parents)
    group = parser.add_argument_group('训练参数')

    group.add_argument("--max_len"       , default=300  , type=str2int  , help="最长句子长度")
    group.add_argument("--beam_size"     , default=5    , type=str2int  , help="单次增加候选句子数量")
    group.add_argument("--alpha"         , default=0.6  , type=str2float, help="alpha")
    group.add_argument("--Compile"       , default=False, type=str2bool , help="是否打开Compile包装")
    group.add_argument("--Greedy"        , default=True , type=str2bool , help='是否启用Greedy评估')
    group.add_argument("--Beam"          , default=True , type=str2bool , help='是否启用Beam评估')
    group.add_argument("--num"           , default=0    , type=str2int  , help='评估模型的编号')
    group.add_argument("--ExpertsFileNum", default=8    , type=str2int  , help='专家文件的数量')
    group.add_argument("--batch_size"    , default=4   , type=str2int  , help="Greedy评估batch大小")
    group.add_argument("--tf32"          , default=False, type=str2bool , help="是否打开脱发2")

    parser.set_defaults(func = function)