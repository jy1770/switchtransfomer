import argparse
from BPE import BPE
from DP import DP
from Function.Function import *

def function(args: argparse.Namespace):
    # 以en-de为例
    # —————————————— # 制作 [Raw-en-de.txt],[TrainBPE-en-de.txt] 放在 /Data/TrainData/
    if args.DP :
        DP.DataProcessing(args) 
    # —————————————— # 制作 [BPE-en-de.model] , [BPE-en-de.vocab]  放在 /Data/TrainData/
    if args.train :
        BPE.train(args) 
    # —————————————— # 制作 [en-de.txt] 放在 /Data/TrainData/
    if args.use :
        BPE.use(args)


def add_subparser(subparsers: argparse._SubParsersAction, parents=None):
    if parents is None:
        parents = []
    parser = subparsers.add_parser('ppc', help='数据处理',parents=parents)
    group = parser.add_argument_group('数据预处理')
    group.add_argument("--DP", default=True, type=str2bool, help="是否制作BPE.model")
    group.add_argument("--train", default=True, type=str2bool, help="是否制作BPE.model")
    group.add_argument("--use"  , default=True, type=str2bool, help="是否制作src.txt,tgt.txt")
    parser.set_defaults(func = function)



