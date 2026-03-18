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

    modelconfig = Config(args, PadId, device)
    model = modelconfig.load_model()

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

    model = model.eval()

    with torch.inference_mode(): # inference_mode 比 no_grad 更快/更省
        for st in tqdm(range(0, len(src_ids_list), args.batch_size),desc=f'正在评估:switchtransfomer_{args.d_model}_{args.SrcName}_{args.TgtName}_{args.num}.pt'):
            batch_ids = src_ids_list[st: st + args.batch_size]
            B = len(batch_ids)
            max_src = max(len(x) for x in batch_ids)

            # ——— src padding -> (B, Lsrc)
            src = torch.full((B, max_src), PadId, device=device, dtype=torch.long)
            for i, ids in enumerate(batch_ids):
                src[i, :len(ids)] = torch.tensor(ids, device=device)

            # ——— Encoder 一次跑完
            src, src_pad_mask = model.Transfomer.forward_Encoder(src, model.ExpertsSet)

            # ——— Greedy 解码
            tgt = torch.full((B, 1), BosId, device=device, dtype=torch.long)
            finished = torch.zeros((B,), device=device, dtype=torch.bool)
            
            for _ in range(args.max_len):
                out = model.Transfomer.forward_Decoder(tgt, src, src_pad_mask, model.ExpertsSet)  # (B, t, V)

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


def length_penalty(length: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    GNMT length penalty:
        lp(l) = ((5 + l) / 6) ^ alpha
    length: shape [...], dtype long/float 都行
    """
    if alpha <= 0:
        return torch.ones_like(length, dtype=torch.float32)
    length = length.to(torch.float32)
    return torch.pow((5.0 + length) / 6.0, alpha)


@torch.inference_mode()
def beam_search_batch(model,src_enc,src_pad_mask,bos_id,eos_id,pad_id,max_len=256,beam_size=4,alpha=0.6):
    """
    AIAIN-style beam search (beam=4, alpha=0.6).
    - uses log-prob sum with GNMT length penalty
    - no coverage penalty, no n-gram blocking, no extra heuristics
    - forbid PAD and BOS during generation
    - once a beam emits EOS, it is marked finished and does not grow further
    """
    device = src_enc.device
    B = src_enc.size(0)

    V = None  # vocab size (infer after first decoder call)

    # 扩展 encoder 输出到 beam 维度： (B, ...) -> (B*beam, ...)
    src_enc_beam = src_enc.repeat_interleave(beam_size, dim=0)
    src_pad_beam = src_pad_mask.repeat_interleave(beam_size, dim=0)

    # beam token 序列（包含 BOS）
    # shape: (B, beam, T)
    tgt = torch.full((B, beam_size, 1), bos_id, device=device, dtype=torch.long)

    # 每个 beam 的“原始累积 logprob”（不做长度惩罚）
    beam_raw = torch.full((B, beam_size), -1e9, device=device, dtype=torch.float32)
    beam_raw[:, 0] = 0.0  # 第 0 条 beam 起始分数为 0

    # 每个 beam 的“有效长度”（不包含 BOS，包含 EOS）
    beam_len = torch.zeros((B, beam_size), device=device, dtype=torch.long)

    # 是否已结束（生成过 EOS）
    finished = torch.zeros((B, beam_size), device=device, dtype=torch.bool)

    # 归一化后的 beam score（用于 topk 选择）
    beam_score = beam_raw / length_penalty(torch.clamp(beam_len, min=1), alpha)

    for step in range(1, max_len + 1):
        # (B, beam, T) -> (B*beam, T)
        tgt_flat = tgt.view(B * beam_size, -1)

        # decoder 输出 logits: (B*beam, T, V)
        out = model.Transfomer.forward_Decoder(tgt_flat, src_enc_beam, src_pad_beam, model.ExpertsSet)
        logits = out[:, -1, :]  # (B*beam, V)

        if V is None:
            V = logits.size(-1)

        # 禁止生成 PAD / BOS（AIAIN beam 里常规做法）
        logits = logits.clone()
        logits[:, pad_id] = -1e9
        logits[:, bos_id] = -1e9

        log_probs = F.log_softmax(logits, dim=-1)  # (B*beam, V)

        # 把已经 finished 的 beam 冻结：只能“生成 PAD”，且 logprob=0（不改变 raw 分数）
        finished_flat = finished.view(B * beam_size)
        if finished_flat.any():
            log_probs[finished_flat] = -1e9
            log_probs[finished_flat, pad_id] = 0.0

        # (B*beam, V) -> (B, beam, V)
        log_probs = log_probs.view(B, beam_size, V)

        # 候选 raw 分数：cand_raw = beam_raw + log_probs
        cand_raw = beam_raw.unsqueeze(-1) + log_probs  # (B, beam, V)

        # 候选长度：未结束的 beam 生成了一个 token -> 长度 = step
        # 已结束 beam 长度保持不变
        cand_len = torch.full((B, beam_size, V), step, device=device, dtype=torch.long)
        if finished.any():
            # finished beams keep their old length (broadcast to V)
            keep_len = beam_len.unsqueeze(-1).expand_as(cand_len)
            cand_len = torch.where(finished.unsqueeze(-1), keep_len, cand_len)

        # 候选 score（用于选择 topk）：cand_score = cand_raw / lp(cand_len)
        cand_score = cand_raw / length_penalty(torch.clamp(cand_len, min=1), alpha)

        # 对每个样本，从 beam*V 个候选里选 top beam_size
        cand_score_flat = cand_score.view(B, -1)  # (B, beam*V)
        top_score, top_idx = torch.topk(cand_score_flat, k=beam_size, dim=-1)

        # top_idx 映射回 (beam_idx, token_idx)
        top_beam = top_idx // V  # (B, beam)
        top_tok = top_idx % V    # (B, beam)

        # 同步拿到对应的 raw 分数和长度
        cand_raw_flat = cand_raw.view(B, -1)
        new_raw = cand_raw_flat.gather(dim=1, index=top_idx)  # (B, beam)

        cand_len_flat = cand_len.view(B, -1)
        new_len = cand_len_flat.gather(dim=1, index=top_idx)  # (B, beam)

        # 更新 finished：继承旧 finished + 新生成 EOS
        old_finished = finished.gather(dim=1, index=top_beam)
        new_finished = old_finished | (top_tok == eos_id)

        # 组装新的 tgt 序列：从旧 beam 里 gather，再 append 新 token
        T = tgt.size(-1)
        gather_index = top_beam.unsqueeze(-1).expand(B, beam_size, T)
        prev_tgt = tgt.gather(dim=1, index=gather_index)  # (B, beam, T)
        tgt = torch.cat([prev_tgt, top_tok.unsqueeze(-1)], dim=-1)  # (B, beam, T+1)

        # 对于“已经 finished 的 beam”，我们不希望长度继续增长
        # new_len 已经处理了 finished 继承旧长度（上面 cand_len 做了）
        beam_raw = new_raw
        beam_len = new_len
        finished = new_finished
        beam_score = top_score

        # 所有样本的所有 beam 都 finished 就可以提前结束
        if finished.all():
            break

    # 最终选每个样本 score 最高的 beam
    best = beam_score.argmax(dim=1)  # (B,)
    best_seq = tgt[torch.arange(B, device=device), best]  # (B, T)

    # 去 BOS，遇到 EOS/PAD 截断
    best_seq = best_seq[:, 1:]  # remove BOS
    out_ids = []
    for i in range(B):
        seq = best_seq[i].tolist()
        cut = []
        for tid in seq:
            if tid == eos_id or tid == pad_id:
                break
            cut.append(tid)
        out_ids.append(cut)

    return out_ids

def Beam(args: argparse.Namespace):
    # —————————————— # 初始化地址参数
    BPEPath  = f'{args.DataPath}Data/TrainData/BPE-{args.SrcName}-{args.TgtName}.model'
    SrcFilePath,RefFilePath = get_FilePath(args)

    # —————————————— # AIAIN beam 设置（强制对齐）
    # Paper commonly uses beam=4, alpha=0.6
    beam_size = 4
    alpha = 0.6

    device, tgt_texts = 'cuda:0', []
    BPE = spm.SentencePieceProcessor(BPEPath)
    BosId, EosId, PadId = BPE.bos_id(), BPE.eos_id(), BPE.pad_id()

    modelconfig = Config(args, PadId, device)
    model = modelconfig.load_model().eval()

    # —————————————— # 获取参考集（AIAIN 通常是 case-sensitive，因此不要 lowercase）
    src_lines = read_sgm_segs(SrcFilePath, lowercase=False)
    ref_texts = read_sgm_segs(RefFilePath, lowercase=False)
    print(f'SrcFilePath : {SrcFilePath}')
    print(f'RefFilePath : {RefFilePath}')

    # —————————————— # 预先编码（不要 lower）
    src_ids_list = []
    for s in src_lines:
        ids = [BosId] + BPE.Encode(s.rstrip()) + [EosId]
        src_ids_list.append(ids)

    with torch.inference_mode():
        for st in tqdm(range(0, len(src_ids_list), args.batch_size),desc=f'Beam评估(beam=4,alpha=0.6): switchtransfomer_{args.d_model}_{args.SrcName}_{args.TgtName}_{args.num}.pt'):
            batch_ids = src_ids_list[st: st + args.batch_size]
            B = len(batch_ids)
            max_src = max(len(x) for x in batch_ids)

            # src padding -> (B, Lsrc)
            src = torch.full((B, max_src), PadId, device=device, dtype=torch.long)
            for i, ids in enumerate(batch_ids):
                src[i, :len(ids)] = torch.tensor(ids, device=device)

            # Encoder
            src_enc, src_pad_mask = model.Transfomer.forward_Encoder(src, model.ExpertsSet)

            # Beam search decode
            batch_out_ids = beam_search_batch(model,src_enc,src_pad_mask,BosId, EosId, PadId,max_len=args.max_len,beam_size=beam_size,alpha=alpha)

            # ids -> text
            for ids in batch_out_ids:
                tgt_texts.append(BPE.DecodeIds(ids))

        torch.cuda.empty_cache()

    # BLEU：尽量贴近 multi-bleu/perl 的 13a tokenization（sacrebleu 默认就是 13a）
    bleu = sacrebleu.corpus_bleu(tgt_texts, [ref_texts], tokenize="13a")
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

    parser.set_defaults(func = function)