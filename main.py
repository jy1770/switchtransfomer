import Preprocessing
import TestSwitchTransfomer
import TrainSwitchTransfomer
import argparse
from Function.Function import *

if __name__ == '__main__':
    # —————————————— # 创建参数接收器
    parent_parser = argparse.ArgumentParser(add_help=False)
    group = parent_parser.add_argument_group('地址参数')

    group.add_argument("--DataPath"  , default='C:/Users/Lenovo/Desktop/bs/Data/'  , type=str  , help="数据库地址")
    group = parent_parser.add_argument_group('基础参数')
    group.add_argument("--SrcName", default='en', type=str, help="原语言名称")
    group.add_argument("--TgtName", default='fr', type=str, help="原句子名称")
    group = parent_parser.add_argument_group('模型参数')
    group.add_argument("--d_model"   , default=512  , type=str2int  , help="模型的维度")
    group.add_argument("--HeadNum"   , default=8    , type=str2int  , help="多头数")
    group.add_argument("--d_ff"      , default=2048 , type=str2int  , help="隐含层维度数")
    group.add_argument("--dropout"   , default=0.1  , type=str2float, help="丢失率")
    group.add_argument("--N"         , default=6    , type=str2int  , help="模型层数")
    group.add_argument("--vocab_size", default=32000, type=str2int  , help="token种类数") # en-fr:32000  en-de:37000
    group.add_argument("--ExpertNum" , default=8    , type=str2int  , help="单层专家数")
    group.add_argument("--capacity_factor", default=1.25 , type=str2float, help="专家容量系数")
    group = parent_parser.add_argument_group('设备参数')
    group.add_argument("--GpuNum"    , default=1    , type=str2int  , help="当前设备有多少GPU")
    
    # —————————————— # 增加子接收器
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommands')
    Preprocessing.add_subparser(subparsers, parents=[parent_parser])
    TrainSwitchTransfomer.add_subparser(subparsers, parents=[parent_parser])
    TestSwitchTransfomer.add_subparser(subparsers, parents=[parent_parser])
    
    # —————————————— # 处理接收的参数
    args = parser.parse_args()
    assert args.ExpertNum % args.GpuNum == 0, "ExpertNum 必须能被 GpuNum 整除"
    args.func(args)
    
# .\\venv\\Scripts\\python.exe .\\SwitchTransfomer\\main.py ppc --DP True --train True --use True # 数据预处理
# screen -dmS test bash -c "./venv/bin/torchrun --nproc-per-node 8 ./SwitchTransfomer/main.py train --DataPath /root/autodl-tmp/Data/ --GpuNum 8" # 训练 RTX 5090 * 8
# .\\venv\\Scripts\\python.exe .\\SwitchTransfomer\\main.py test --Greedy True --Beam False --num 10000 --GpuNum 1 --batch_size 2 # 测试 RTX 2080Ti * 4