import torch.nn as nn

class SwitchTransfomer(nn.Module):
    def __init__(self,transfomer,expertsset):
        super().__init__()
        self.Transfomer = transfomer
        self.ExpertsSet = expertsset

    def forward(self,src,tgt):
        return self.Transfomer(src,tgt,self.ExpertsSet)

