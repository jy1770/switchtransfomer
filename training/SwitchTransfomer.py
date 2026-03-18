import torch.nn as nn

class SwitchTransfomer(nn.Module):
    def __init__(self,transfomer,expertsset):
        super().__init__()
        self.transfomer = transfomer
        self.expertset = expertsset

    def forward(self,src,tgt):
        return self.transfomer(src,tgt,self.expertset)