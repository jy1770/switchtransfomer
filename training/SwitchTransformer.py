import torch.nn as nn

class SwitchTransformer(nn.Module):
    def __init__(self,transformer,expertset):
        super().__init__()
        self.transformer = transformer
        self.expertset = expertset

    def forward(self,src,tgt):
        return self.transformer(src,tgt,self.expertset)