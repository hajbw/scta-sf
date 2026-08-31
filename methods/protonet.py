import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .template import MetaTemplate


class ProtoNet(MetaTemplate):#继承metatemplate，对三个抽象方法重写
    def __init__(self, params, model_func, n_way, n_support):
        super(ProtoNet, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()#交叉熵损失
        self.avgpool = nn.AdaptiveAvgPool2d(1)#自适应平均池化

    def feature_forward(self, x):#重写
        out = self.avgpool(x).view(x.size(0),-1)#平均池化
        return out

    def set_forward(self, x, is_feature=False):#返回query和原型的距离
        z_support, z_query = self.parse_feature(x, is_feature)#template里面的函数，返回特征
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)#计算原型：均值向量
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        scores = self.euclidean_dist(z_query, z_proto)#调用欧几里得距离
        return scores#距离

    def set_forward_loss(self, x):#返回：准确率，标签长度，损失函数的返回值，距离
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))#查询集的图像，先建立空的
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.n_way), self.n_query)#查询集的标签
        scores = self.set_forward(x)#前向传播，得到距离
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)#topk：取最大最小值；top-1中取最大值，按行，排序
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)#准确率：预测和真实相等，就加上

        return float(top1_correct), len(y_label), self.loss_fn(scores, y_query), scores

    def euclidean_dist(self, x, y):#计算X和Y的欧氏距离
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        score = -torch.pow(x - y, 2).sum(2)
        return score
