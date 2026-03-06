import hashlib
import argparse
from tqdm import tqdm
from pathlib import Path
'''
                                en-de en-fr
training-parallel-europarl-v7 |   *  |  *
training-parallel-commoncrawl |   *  |  *
training-parallel-un          |      |  *
training-giga-fren            |   *  |  *
training-parallel-nc-v9       |      |  *

          [Raw]      [去重+长度过滤]
[en-de] : 4521186    4293875      
[en-fr] : 40842838   34593021                
'''
def key(sen : str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(sen.encode("utf-8"))
    return h.digest()

def DataProcessing(args: argparse.Namespace):
    # —————————————— # 初始化参数
    FileNameSet,SrcSeen,SenSeen = set(),set(),set()
    folder        = Path(f'{args.DataPath}RawData/RawTrainData/')
    RawSrcTgtPath = Path(f'{args.DataPath}Data/TrainData/Raw-{args.SrcName}-{args.TgtName}.txt')
    TrainBPE_SrcTgtPath = Path(f'{args.DataPath}Data/TrainData/TrainBPE-{args.SrcName}-{args.TgtName}.txt')

    # —————————————— # 合并文件并写入文件
    num=0
    with RawSrcTgtPath.open("w", encoding="utf-8", newline="\n") as SrcTgt , TrainBPE_SrcTgtPath.open("w", encoding="utf-8", newline="\n") as TrainBPE:
        for p in folder.iterdir() :
            if p.is_file() and (args.SrcName in p.name) and (args.TgtName in p.name) and ('annotation' not in p.name) and (p.stem not in FileNameSet):
                    SrcPath = Path(f'{args.DataPath}RawData/RawTrainData/{p.stem}.{args.SrcName}')
                    TgtPath = Path(f'{args.DataPath}RawData/RawTrainData/{p.stem}.{args.TgtName}')
                    FileNameSet.add(p.stem)

                    with SrcPath.open("r", encoding="utf-8", errors="replace") as Src , TgtPath.open("r", encoding="utf-8", errors="replace") as Tgt:
                        for SrcUnit,TgtUnit in tqdm(zip(Src,Tgt) , desc=f'正在读取{p.stem}'):
                            # —————— # 初始参数
                            SrcUnit = SrcUnit.rstrip().lower() ; TgtUnit = TgtUnit.rstrip().lower()
                            SrcUnitLen = len(SrcUnit.split())  ; TgtUnitLen = len(TgtUnit.split())

                            # —————— # 长度过滤
                            MidMax = max(SrcUnitLen, TgtUnitLen) ; MidMin = min(SrcUnitLen, TgtUnitLen)
                            if (MidMax>250) or (MidMin<1) or (MidMax/MidMin>3):
                                continue

                            # —————— # 去重
                            SrcKey = key(SrcUnit) ; SenKey = key(SrcUnit + TgtUnit)
                            if (SrcKey in SrcSeen) or (SenKey in SenSeen):
                                continue
                            SrcSeen.add(SrcKey)   ; SenSeen.add(SenKey)

                            # —————— # 写入文件
                            num+=1
                            SrcTgt.write(SrcUnit + "\t" + TgtUnit + "\n")
                            TrainBPE.write(SrcUnit + "\n" + TgtUnit + "\n")
    del SrcSeen
    del SenSeen 
    print(f'总句对 : {num}')