import torch
import torch.nn as nn

class Transfomer(nn.Module):
    def __init__(self, encoder,decoder,PadId,h):
        super().__init__()
        self.Encoder = encoder
        self.Decoder = decoder
        self.PadId   = PadId
        self.h = h
    def make_src_mask(self, src):
        return src == self.PadId
    def make_tgt_mask(self, tgt):
        B  , L = tgt.shape[0] , tgt.shape[1]
        tgt_pad_mask = (tgt != self.PadId).unsqueeze(1)
        tgt_sub_mask = torch.tril(torch.ones((L,L), device='cuda')).bool()
        tgt_mask = ~ (tgt_pad_mask & tgt_sub_mask)
        return tgt_mask.unsqueeze(1).expand(B, self.h, L, L).reshape(B*self.h, L, L) , tgt == self.PadId
    def forward(self, src, tgt,expertsset):
        # —————————————— # 制作遮罩
        src_pad_mask          = self.make_src_mask(src)
        tgt_mask,tgt_pad_mask = self.make_tgt_mask(tgt)
        # —————————————— # 前向传播
        src,src_lb_loss,src_z_loss = self.Encoder(src,src_pad_mask,expertsset.src)
        output,tgt_lb_loss,tgt_z_loss = self.Decoder(tgt,src,tgt_mask,tgt_pad_mask,src_pad_mask,expertsset.tgt)
        return output,(src_lb_loss+tgt_lb_loss)/2,(src_z_loss+tgt_z_loss) / 2