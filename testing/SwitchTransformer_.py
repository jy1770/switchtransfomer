import torch.nn as nn

class SwitchTransformer(nn.Module):
    def __init__(self,transformer,expertsset):
        super().__init__()
        self.Transformer = transformer
        self.ExpertsSet = expertsset

    def forward(self,src,tgt):
        return self.Transformer(src,tgt,self.ExpertsSet)

