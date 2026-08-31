'''
@file: bdc_modele.py
@author: Fei Long
@author: Jiaming Lv
Please cite the paper below if you use the code:

Jiangtao Xie, Fei Long, Jiaming Lv, Qilong Wang and Peihua Li. Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification. IEEE Int. Conf. on Computer Vision and Pattern Recognition (CVPR), 2022.

Copyright (C) 2022 Fei Long and Jiaming Lv

All rights reserved.
'''

import torch
import torch.nn as nn

class BDC(nn.Module):
    def __init__(self, is_vec=True, input_dim=640, dimension_reduction=None, activate='relu'):
        super(BDC, self).__init__()
        self.is_vec = is_vec#默认是true
        self.dr = dimension_reduction#降维，128
        self.activate = activate#激活函数relu
        self.input_dim = input_dim[0]#输入维度，input_dim为[640, 10, 10]，[通道数，特征图的高，特征图的宽]
        if self.dr is not None and self.dr != self.input_dim:#如果降维且不等于输入维度
            if activate == 'relu':#激活函数
                self.act = nn.ReLU(inplace=True)
            elif activate == 'leaky_relu':#leaky——relu激活
                self.act = nn.LeakyReLU(0.1)
            else:
                self.act = nn.ReLU(inplace=True)

            self.conv_dr_block = nn.Sequential(#卷积维度下降块：一个卷积，一个批量归一化，一个激活
            nn.Conv2d(self.input_dim, self.dr, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.dr),
            self.act
            )
        output_dim = self.dr if self.dr else self.input_dim#有降维就输出降维，没有就输出输入维度 128
        if self.is_vec:#默认是true
            self.output_dim = int(output_dim*(output_dim+1)/2)#输出维度，没懂
        else:
            self.output_dim = int(output_dim*output_dim)
        #温度系数
        self.temperature = nn.Parameter(torch.log((1. / (2 * input_dim[1]*input_dim[2])) * torch.ones(1,1)), requires_grad=True)
        #这个参数是论文中所指的温度参数，实际使用时，我们把它加到了BDC里计算特征的两两欧氏距离中。
        #由于在求解BDC矩阵时算完特征之间的欧氏距离后，剩下的运算均是线性的，故实际上可认为加了一个具有一定初始值的温度在logits上。
        self._init_weight()#初始化权重

    def _init_weight(self):#初始化权重
        for m in self.modules():#遍历模型中的层
            if isinstance(m, nn.Conv2d):#如果是卷积层
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')#kaiming初始化，fan——out
            elif isinstance(m, nn.BatchNorm2d):#如果是批量归一化层
                nn.init.constant_(m.weight, 1)#用于将权重或偏置初始化为常数值
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):#前向传播
        if self.dr is not None and self.dr != self.input_dim:#需要降维
            x = self.conv_dr_block(x)#卷积维度下降块：一个卷积，一个批量归一化，一个激活,x:batchsize,640,10,10---batchsize,128,10,10
        x = BDCovpool(x, self.temperature)#BDC池化，返回BDC矩阵8,128,128
        if self.is_vec:#默认是true
            x = Triuvec(x)#提取上三角，变成向量8,8256
            #x = reduce_pooling(x, self.batch_lengths)
        else:
            x = x.reshape(x.shape[0], -1)#否则变成按x的行生成
        return x#BDC向量
'''
def reduce_pooling(x, batch_lengths):
    x = x.transpose(1, 0)
    averaged_features = []
    i0 = 0
    for b_i, length in enumerate(batch_lengths):
        # Average features for each batch cloud
        averaged_features.append(torch.mean(x[i0:i0 + length], dim=0))

        # Increment for next cloud
        i0 += length
    x = torch.stack(averaged_features)
    x = x.transpose(1, 0)  # 64,688
    return x
'''
def BDCovpool(x, t):#BDC池化，传入x和温度，返回BDC矩阵
    batchSize, dim, h, w = x.data.shape#批量size，维度，高，宽batchsize,128,10,10
    M = h * w#面积=高x宽100
    x = x.reshape(batchSize, dim, M)#8,128,100
    #eye：创建对角矩阵，dim*dim的，devic是gpu，，1行dim列dim高，，重复，从后往前看，复制了【【】】batchsize次，dtype元素的数据类型
    I = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(x.dtype)#8,128,128
    I_M = torch.ones(batchSize, dim, dim, device=x.device).type(x.dtype)#全是1的矩阵8,128,128
    #transpose：转置矩阵，x_pow2是x乘x的转置
    x_pow2 = x.bmm(x.transpose(1, 2))#bmm：如果input是一个(b, n , m)的张量，mat2是一个(b, m, p)张量，则输出形状为(b, n, p)8,128,128
    #三个算子的A：I是单位矩阵，I_M是1，第二项是转置，sym体现在转置上
    dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2#{A}=2{(1(X^TX 0 I))}sym-2X^TX
    #最小值是08,128,128
    dcov = torch.clamp(dcov, min=0.0)#clamp：将输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量
    dcov = torch.exp(t)* dcov#e^温度*矩阵
    dcov = torch.sqrt(dcov + 1e-5)#开根号8,128,128
    t = dcov - 1. / dim * dcov.bmm(I_M) - 1. / dim * I_M.bmm(dcov) + 1. / (dim * dim) * I_M.bmm(dcov).bmm(I_M)
    #t={A}-2/d{(1\{A})}_{sym}+2/d^2 {A}1 8,128,128
    return t


def Triuvec(x):#提取x的上三角
    batchSize, dim, dim = x.shape#batch，维度，维度8,128,128
    r = x.reshape(batchSize, dim * dim)#batch行，维度*维度列8,16384
    #triu：提取上三角
    I = torch.ones(dim, dim).triu().reshape(dim * dim)#I是dim*dim的矩阵，只有上三角全是1 16364
    index = I.nonzero(as_tuple = False)#用于输出数组的非零值的索引，as_tuple：如果设为False，则返回一个二维张量，其中每一行都是非零值的索引 8256,1
    y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)#8,8256
    y = r[:, index].squeeze()#squeeze：从张量形状中移除大小为 1 的维度。
   # y：x按照上三角提取，删除元素是1的维度，返回张量
    return y
