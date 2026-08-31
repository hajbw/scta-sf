import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads=8, head_dim_ratio=1., qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = round(dim // num_heads * head_dim_ratio)
        self.head_dim = head_dim
        # self.scale = qk_scale or head_dim ** -0.5
        #new qk_scale to avoid NAN when using amp.
        qk_scale_factor = qk_scale if qk_scale is not None else -0.25
        self.scale = head_dim ** qk_scale_factor

        self.qkv = nn.Conv2d(dim, head_dim * num_heads * 3, 1, stride=1, padding=0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(self.head_dim * self.num_heads, dim, 1, stride=1, padding=0, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)
        #self.norm = nn.LayerNorm([640, 10, 10])

    def forward(self, x_i):
        B, C, H, W = x_i.shape#torch.Size([25, 640, 10, 10])
        x = self.qkv(x_i)
        #torch.Size([3, 25, 8, 100, 80])
        qkv = rearrange(x, 'b (x y z) h w -> x b y (h w) z', x=3, y=self.num_heads, z=self.head_dim)#torch.Size([3, 5, 3, 196, 64])
        # changed by wentao to add a semantic prompt
        if H != W:
            qkv = qkv[:, :, :, :(H-1)*W+1]#torch.Size([3, 5, 3, 56, 128])  torch.Size([3, 5, 3, 50, 128])
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ( (q * self.scale) @ (k.transpose(-2,-1) * self.scale) )
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        outx = attn @ v

        x = rearrange(outx, 'b y (h w) z -> b (y z) h w', h=H, w=W)#torch.Size([5, 192, 14, 14])
        x = self.proj(x)
        x = self.proj_drop(x)
        #out = self.norm(x)
        return x+x_i


class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Conv2d(dim, dim // 8, 1)
        self.key = nn.Conv2d(dim, dim // 8, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm([640, 10, 10])
        self.conv1 = ConvBlock(100, 36, 1)
        self.conv2 = nn.Conv2d(36, 100, 1, stride=1, padding=0)


    def forward(self, x_identity, y_identity):
        B1, C, H, W = x_identity.shape#25,640,10,10
        B2 = y_identity.shape[0]#80

        x=x_identity
        y=y_identity
        #x = self.norm(x_identity)
        #y = self.norm(y_identity)
        for scale_i in range(1,10):
            scale = (C // 8) ** (scale_i*0.1)
            qx = self.query(x).view(B1, -1, H * W).permute(0, 2, 1) * scale  # B, H*W, C'5,100,80
            ky = self.key(y).view(B2, -1, H * W) * scale  # B, C', H*W 80,80,100
            vy = self.value(y).view(B2, -1, H * W)  # B, C, H*W 80,640,100

            qx = qx.reshape(5,-1,*qx.shape[1:]).unsqueeze(2)#torch.Size([5, 1, 1, 100, 80])
            ky = ky.reshape(5,-1,*ky.shape[1:]).unsqueeze(1)#torch.Size([5, 1, 16, 80, 100])
            vy = vy.reshape(5,-1,*vy.shape[1:]).unsqueeze(1).repeat(1, int(B1/5), 1,1,1)  # shengwei,torch.Size([5, 1, 16, 640, 100])
            attnx = torch.matmul(qx, ky)
            attnx_softmax = self.softmax(attnx)  # B, H*W, H*W 5,1,16,100,100
            attnx_var = torch.var(attnx_softmax)
            if attnx_var > 1e-5:#1e-5:
                break

        outx = torch.matmul(attnx_softmax, vy.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)#5,5,16,640,100
        #max
        outx = torch.max(outx, dim=1)[0]
        outx = outx.view(B2, C, H, W)  # B, C, H, W
        outx = outx.permute(0,2,3,1)
        outx = self.mlp(outx)  # Apply MLP and permute back
        outx = outx.permute((0,3,1,2))
        #outx = self.norm(outx)  # Apply normalization

        qy = self.query(y).view(B2, -1, H * W).permute(0, 2, 1) * scale  # B, H*W, C'80,100,80
        kx = self.key(x).view(B1, -1, H * W) * scale  # B, C', H*W5,8,100
        vx = self.value(x).view(B1, -1, H * W)  # B, C, H*W5,640,100

        qy = qy.reshape(5, -1, *qy.shape[1:]).unsqueeze(2)  # ([5, 16, 1, 100, 80])
        kx = kx.reshape(5, -1, *kx.shape[1:]).unsqueeze(1)  # ([5, 1, 1, 80, 100])
        vx = vx.reshape(5, -1, *vx.shape[1:]).unsqueeze(1).repeat(1, int(B2 / 5), 1, 1, 1)  # shengwei,([5, 16, 1, 640, 100])
        attny = torch.matmul(qy, kx)

        attny_softmax = self.softmax(attny)  # B, H*W, H*W 5,16,1,100,100

        outy = torch.matmul(attny_softmax, vx.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)#torch.Size([5, 16, 1, 640, 100])

        outy = torch.max(outy, axis=1)[0]
        outy = outy.view(B1, C, H, W)  # B, C, H, W 80,640,10,10
        outy = self.mlp(outy.permute(0,2,3,1)).permute(0,3,1,2)  # Apply MLP and permute back
        #outy = self.norm(outy)  # Apply normalization

        #add resdual
        outx = outx*0.4+y_identity
        outy = outy*0.4+x_identity

        return outy, outx

