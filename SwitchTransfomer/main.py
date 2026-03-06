import Preprocessing
import TestSwitchTransfomer
import TrainSwitchTransfomer
import argparse
from Function.Function import *

if __name__ == '__main__':
    # —————————————— # 创建参数接收器
    parent_parser = argparse.ArgumentParser(add_help=False)
    group = parent_parser.add_argument_group('地址参数')
    group.add_argument("--DataPath"  , default='C:/Users/Lenovo/Desktop/python/Plural/Data/'  , type=str  , help="数据库地址")
    group = parent_parser.add_argument_group('基础参数')
    group.add_argument("--SrcName", default='en', type=str, help="原语言名称")
    group.add_argument("--TgtName", default='fr', type=str, help="原句子名称")
    group = parent_parser.add_argument_group('模型参数')
    group.add_argument("--d_model"   , default=512  , type=str2int  , help="模型的维度")
    group.add_argument("--HeadNum"   , default=8    , type=str2int  , help="多头数")
    group.add_argument("--d_ff"      , default=2048 , type=str2int  , help="隐含层维度数")
    group.add_argument("--dropout"   , default=0.1  , type=str2float, help="丢失率")
    group.add_argument("--N"         , default=6    , type=str2int  , help="模型层数")
    group.add_argument("--vocab_size", default=32000, type=str2int  , help="token种类数") # en-fr
    # group.add_argument("--vocab_size", default=37000, type=str2int  , help="token种类数") # en-de
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

    # —————————————— # 设定默认参数
    base = ['--d_model','512' ,'--d_ff','2048','--HeadNum','8' ,'--dropout','0.1']
    big  = ['--d_model','1024','--d_ff','4096','--HeadNum','16','--dropout','0.3']
    Linux= ['train','--DataPath','/root/autodl-tmp/Data/','--GpuNum','8']
    
    # ———————————————— # 执行【下面这4行代码不是固定的，会根据任务和在不同的设备修改，我写在这是因为方便运行】
    # args = parser.parse_args(['ppc','--DP','False','--train','False','--use','True'])  # 数据初始化
    # args = parser.parse_args(['test','--Greedy','True','--Beam','False','--num','4','--GpuNum','1',"--batch_size",'4']) # 本地测试模型
    # args = parser.parse_args(['test','--DataPath','/root/autodl-tmp/Data/','--Greedy','True','--Beam','False','--num','2','--GpuNum','4',"--batch_size",'32']) # 服务器测试模型
    args = parser.parse_args(Linux + base) # 训练模型

    assert args.ExpertNum % args.GpuNum == 0, "ExpertNum 必须能被 GpuNum 整除"

    args.func(args)

# screen -dmS test bash -c "./venv/bin/torchrun --nproc-per-node 8 ./SwitchTransfomer/main.py" # 训练 RTX 5090 * 8
# ./venv/bin/python ./SwitchTransfomer/main.py # 测试 RTX 2080Ti * 4