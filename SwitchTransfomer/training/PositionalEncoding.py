import math
import torch
import torch.nn as nn

# —————————————— # 位置嵌入
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)# 初始化 dropout 层
        # 计算位置编码
        pe = torch.zeros(max_len, d_model)                # 创建一个全零矩阵[max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)  # 创建位置索引，形状为 [max_len, 1]
        div_term = torch.exp(torch.arange(0,d_model,2)*-(math.log(10000.0)/d_model))# 计算归一化因子

        pe[:, 0::2] = torch.sin(position * div_term)  # 对偶数索引位置进行 sin 操作
        pe[:, 1::2] = torch.cos(position * div_term)  # 对奇数索引位置进行 cos 操作
        pe = pe.unsqueeze(0)            # 增加一个维度，使得形状变为 [1, max_len, d_model]
        self.register_buffer("pe", pe)  # 将位置编码注册为 buffer，防止梯度计算

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False) # 将位置编码加到输入张量上，并关闭位置编码的梯度计算
        return self.dropout(x)  # 返回应用了 dropout 的输出