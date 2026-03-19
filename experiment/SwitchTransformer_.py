import torch.nn as nn

class SwitchTransformer(nn.Module):
    def __init__(self,transformer,expertsset):
        super().__init__()
        self.transformer = transformer
        self.expertset = expertsset

    def forward(self,src,tgt):
        return self.transformer(src,tgt,self.expertset)

