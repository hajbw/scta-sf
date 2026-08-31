import numpy as np
import torch
from torch import nn
from torch.nn import init


def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x


def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x


class SplitAttention(nn.Module):
    def __init__(self, channel=512, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)  # bs,k,n,c 85，3，100，640
        a = torch.sum(torch.sum(x_all, 1), 1)  # bs,c 85，640
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  # bs,kc 85，1920
        hat_a = hat_a.reshape(b, self.k, c)  # bs,k,c 85，3，640
        bar_a = self.softmax(hat_a)  # bs,k,c
        attention = bar_a.unsqueeze(-2)  # #bs,k,1,c 85，3，1，640
        out = attention * x_all  # #bs,k,n,c 85，3，100，640
        out = torch.sum(out, 1).reshape(b, h, w, c)#85,10,10,640
        return out


class S2Attention(nn.Module):

    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention(channel=channels)

    def forward(self, x):
        b, c, w, h = x.size()#85,640,10,10
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)#85,10,10,1920
        x1 = spatial_shift1(x[:, :, :, :c])#85,10,10,640
        x2 = spatial_shift2(x[:, :, :, c:c * 2])#85,10,10,640
        x3 = x[:, :, :, c * 2:]#85,10,10,640
        x_all = torch.stack([x1, x2, x3], 1)#85,10,10,1920
        a = self.split_attention(x_all)#85,10,10,640
        x = self.mlp2(a)
        x = x.permute(0, 3, 1, 2)#85，640，10，10
        return x


'''
#   输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    s2att = S2Attention(channels=512)
    output = s2att(input)
    print(output.shape)
'''