import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .template import MetaTemplate, MetaTemplate2
from torch.distributions import MultivariateNormal
from .ddwm_utils import Distribution_fitting_with_DDWM, normalize_l2

class ProtoNet(MetaTemplate):#继承metatemplate，对三个抽象方法重写
    def __init__(self, params, model_func, n_way, n_support, noaug=False):
        super(ProtoNet, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()#交叉熵损失
        self.avgpool = nn.AdaptiveAvgPool2d(1)#自适应平均池化
        self.noaug = noaug
        # 去BDC生成BDC向量
        self.n_way = n_way
        self.n_support = n_support

    def feature_forward(self, x):#重写
        out = self.avgpool(x).view(x.size(0),-1)#平均池化
        return out

    def set_forward(self, x, is_feature=False):#返回query和原型的距离
        z_support, z_query = self.parse_feature(x, is_feature)#template里面的函数，返回特征
        # z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)#计算原型：均值向量

        if self.noaug or self.training:

            z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        else:
            n_lsamples = z_support.shape[1]


            num_sampled = int(self.n_aug / self.n_shot)

            with ((torch.no_grad())):
                mean_tch, cov_tch = Distribution_fitting_with_DDWM(z_support, self.base_means, self.base_means_matrix,
                                                                 self.base_cov, alpha=self.dc_alpha, k=self.dc_k)
            samps_at_a_time = 1
            with torch.no_grad():
                sampled_data_lst = []
                mvn_gen = MultivariateNormal(mean_tch, covariance_matrix=cov_tch)
                for _ in range(int(np.ceil(float(num_sampled) / samps_at_a_time))):
                    mvn_gen.sample()
                    norm_samps_tch = mvn_gen.sample((samps_at_a_time, ))
                    norm_samps_tch = torch.as_tensor(norm_samps_tch)
                    sampled_data_lst.append(norm_samps_tch)
                sampled_data = torch.cat(sampled_data_lst, dim=0)[:num_sampled]
                sampled_data = sampled_data.permute(1, 2, 0, 3).cuda()


            with torch.no_grad():
                sampled_data = sampled_data.reshape(z_support.shape[0], n_lsamples*num_sampled, z_support.shape[2])
                #z_support = torch.tensor(z_support)
                X_aug = torch.cat([z_support, sampled_data], dim=-2)

            X_aug_proto = X_aug.contiguous().view(self.n_way, n_lsamples + n_lsamples*num_sampled, -1)
            z_proto = X_aug_proto.mean(1)#支持集原型是均值
            z_query = torch.as_tensor(z_query)

        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        scores = self.euclidean_dist(z_query, z_proto)#调用欧几里得距离
        return scores#距离

    def sample_data(self, x, is_feature=False):  # 前向传播，返回距离
        # initial
        z_support, z_query = self.parse_feature(x, is_feature)  # 返回支持集和查询集特征
        z_support = z_support.reshape(self.n_way, self.n_support, -1)  # 5,16,1,2064


        #add gussi sampler for val set

        if 1==0:

            z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        else:
            n_lsamples = z_support.shape[1]
            '''
            n_usamples = z_query.shape[1]
            n_samples = n_lsamples + n_usamples
            '''
            '''     # norm：p=2，求L-2范数，L-2规范化
            qry_norm = torch.norm(z_query, p=2, dim=1).unsqueeze(1).expand_as(z_query)
            spt_norm = torch.norm(z_support, p=2, dim=1).unsqueeze(1).expand_as(z_support)
            qry_normalized = z_query.div(qry_norm + 1e-6)
            spt_normalized = z_support.div(spt_norm + 1e-6)

            z_query = qry_normalized  # 没有梯度
            z_support = spt_normalized
            '''


            num_sampled = int(self.n_aug / self.n_shot)

            with ((torch.no_grad())):
                mean_tch, cov_tch = Distribution_fitting_with_DDWM(z_support, self.base_means, self.base_means_matrix,
                                                                 self.base_cov, alpha=self.dc_alpha, k=self.dc_k)
            samps_at_a_time = 1
            with torch.no_grad():
                sampled_data_lst = []
                mvn_gen = MultivariateNormal(mean_tch, covariance_matrix=cov_tch)
                for _ in range(int(np.ceil(float(num_sampled) / samps_at_a_time))):
                    mvn_gen.sample()
                    norm_samps_tch = mvn_gen.sample((samps_at_a_time, ))
                    norm_samps_tch = torch.as_tensor(norm_samps_tch)
                    sampled_data_lst.append(norm_samps_tch)
                sampled_data = torch.cat(sampled_data_lst, dim=0)[:num_sampled]
                sampled_data = sampled_data.permute(1, 2, 0, 3).cuda()


            with torch.no_grad():
                sampled_data = sampled_data.reshape(z_support.shape[0], n_lsamples*num_sampled, z_support.shape[2])
                #z_support = torch.tensor(z_support)
                # X_aug = torch.cat([z_support, sampled_data], dim=-2)

            # X_aug_proto = X_aug.contiguous().view(self.n_way, n_lsamples + n_lsamples*num_sampled, -1)
            # z_proto = X_aug_proto.mean(1)#支持集原型是均值
            # z_query = torch.as_tensor(z_query)
        # z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # scores = self.metric(z_query, z_proto)#计算距离

        proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1).view(self.n_way * self.n_support, -1)
        query = torch.as_tensor(z_query).contiguous().view(self.n_way * self.n_query, -1)
        aug = sampled_data.contiguous().view(self.n_way * n_lsamples * num_sampled, -1)

        return proto, query, aug, int(n_lsamples*num_sampled)

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


class ProtoNet_noattn(MetaTemplate2):#继承metatemplate，对三个抽象方法重写
    def __init__(self, params, model_func, n_way, n_support, noaug=False):
        super(ProtoNet_noattn, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()#交叉熵损失
        self.avgpool = nn.AdaptiveAvgPool2d(1)#自适应平均池化
        self.noaug = noaug
        # 去BDC生成BDC向量
        self.n_way = n_way
        self.n_support = n_support

    def feature_forward(self, x):#重写
        out = self.avgpool(x).view(x.size(0),-1)#平均池化
        return out

    def set_forward(self, x, is_feature=False):#返回query和原型的距离
        z_support, z_query = self.parse_feature(x, is_feature)#template里面的函数，返回特征
        # z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)#计算原型：均值向量

        if self.noaug or self.training:

            z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        else:
            n_lsamples = z_support.shape[1]


            num_sampled = int(self.n_aug / self.n_shot)

            with ((torch.no_grad())):
                mean_tch, cov_tch = Distribution_fitting_with_DDWM(z_support, self.base_means, self.base_means_matrix,
                                                                 self.base_cov, alpha=self.dc_alpha, k=self.dc_k)
            samps_at_a_time = 1
            with torch.no_grad():
                sampled_data_lst = []
                mvn_gen = MultivariateNormal(mean_tch, covariance_matrix=cov_tch)
                for _ in range(int(np.ceil(float(num_sampled) / samps_at_a_time))):
                    mvn_gen.sample()
                    norm_samps_tch = mvn_gen.sample((samps_at_a_time, ))
                    norm_samps_tch = torch.as_tensor(norm_samps_tch)
                    sampled_data_lst.append(norm_samps_tch)
                sampled_data = torch.cat(sampled_data_lst, dim=0)[:num_sampled]
                sampled_data = sampled_data.permute(1, 2, 0, 3).cuda()


            with torch.no_grad():
                sampled_data = sampled_data.reshape(z_support.shape[0], n_lsamples*num_sampled, z_support.shape[2])
                #z_support = torch.tensor(z_support)
                X_aug = torch.cat([z_support, sampled_data], dim=-2)

            X_aug_proto = X_aug.contiguous().view(self.n_way, n_lsamples + n_lsamples*num_sampled, -1)
            z_proto = X_aug_proto.mean(1)#支持集原型是均值
            z_query = torch.as_tensor(z_query)

        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        scores = self.euclidean_dist(z_query, z_proto)#调用欧几里得距离
        return scores#距离

    def sample_data(self, x, is_feature=False):  # 前向传播，返回距离
        # initial
        z_support, z_query = self.parse_feature(x, is_feature)  # 返回支持集和查询集特征
        z_support = z_support.reshape(self.n_way, self.n_support, -1)  # 5,16,1,2064


        #add gussi sampler for val set

        if 1==0:

            z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        else:
            # n_lsamples = z_support.shape[1]
            '''
            n_usamples = z_query.shape[1]
            n_samples = n_lsamples + n_usamples
            '''
            '''     # norm：p=2，求L-2范数，L-2规范化
            qry_norm = torch.norm(z_query, p=2, dim=1).unsqueeze(1).expand_as(z_query)
            spt_norm = torch.norm(z_support, p=2, dim=1).unsqueeze(1).expand_as(z_support)
            qry_normalized = z_query.div(qry_norm + 1e-6)
            spt_normalized = z_support.div(spt_norm + 1e-6)

            z_query = qry_normalized  # 没有梯度
            z_support = spt_normalized
            '''


            # num_sampled = int(self.n_aug / self.n_support)

            # print(z_support.shape)

            with ((torch.no_grad())):
                mean_tch, cov_tch = Distribution_fitting_with_DDWM(z_support.contiguous().view(self.n_way, self.n_support, -1).mean(dim=1,keepdim=True), self.base_means, self.base_means_matrix,
                                                                 self.base_cov, alpha=self.dc_alpha, k=self.dc_k)
            samps_at_a_time = 1
            with torch.no_grad():
                sampled_data_lst = []
                mvn_gen = MultivariateNormal(mean_tch, covariance_matrix=cov_tch)
                for _ in range(int(np.ceil(float(self.n_aug) / samps_at_a_time))):
                    mvn_gen.sample()
                    norm_samps_tch = mvn_gen.sample((samps_at_a_time, ))
                    norm_samps_tch = torch.as_tensor(norm_samps_tch)
                    sampled_data_lst.append(norm_samps_tch)
                sampled_data = torch.cat(sampled_data_lst, dim=0)[:self.n_aug]
                sampled_data = sampled_data.permute(1, 2, 0, 3).cuda()


            with torch.no_grad():
                sampled_data = sampled_data.reshape(z_support.shape[0], self.n_aug, z_support.shape[2])
                #z_support = torch.tensor(z_support)
                # X_aug = torch.cat([z_support, sampled_data], dim=-2)

            # X_aug_proto = X_aug.contiguous().view(self.n_way, n_lsamples + self.n_aug, -1)
            # z_proto = X_aug_proto.mean(1)#支持集原型是均值
            # z_query = torch.as_tensor(z_query)
        # z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # scores = self.metric(z_query, z_proto)#计算距离

        proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1).view(self.n_way, -1)
        query = torch.as_tensor(z_query).contiguous().view(self.n_way * self.n_query, -1)
        aug = sampled_data.contiguous().view(self.n_way * self.n_aug, -1)

        # print(z_support.shape)

        # print('proto.shape', proto.shape, 'query.shape', query.shape, 'aug.shape', aug.shape, 'n_aug', int(self.n_aug))

        return proto, query, aug, int(self.n_aug)

    def sample_data_with_support(self, x, is_feature=False):  # 前向传播，返回距离
        # initial
        z_support, z_query = self.parse_feature(x, is_feature)  # 返回支持集和查询集特征
        z_support = z_support.reshape(self.n_way, self.n_support, -1)  # 5,16,1,2064


        #add gussi sampler for val set

        if 1==0:

            z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        else:
            # n_lsamples = z_support.shape[1]
            '''
            n_usamples = z_query.shape[1]
            n_samples = n_lsamples + n_usamples
            '''
            '''     # norm：p=2，求L-2范数，L-2规范化
            qry_norm = torch.norm(z_query, p=2, dim=1).unsqueeze(1).expand_as(z_query)
            spt_norm = torch.norm(z_support, p=2, dim=1).unsqueeze(1).expand_as(z_support)
            qry_normalized = z_query.div(qry_norm + 1e-6)
            spt_normalized = z_support.div(spt_norm + 1e-6)

            z_query = qry_normalized  # 没有梯度
            z_support = spt_normalized
            '''


            # num_sampled = int(self.n_aug / self.n_shot)

            with ((torch.no_grad())):
                mean_tch, cov_tch = Distribution_fitting_with_DDWM(z_support.contiguous().view(self.n_way, self.n_support, -1).mean(dim=1,keepdim=True), self.base_means, self.base_means_matrix,
                                                                 self.base_cov, alpha=self.dc_alpha, k=self.dc_k)
            samps_at_a_time = 1
            with torch.no_grad():
                sampled_data_lst = []
                mvn_gen = MultivariateNormal(mean_tch, covariance_matrix=cov_tch)
                for _ in range(int(np.ceil(float(self.n_aug) / samps_at_a_time))):
                    mvn_gen.sample()
                    norm_samps_tch = mvn_gen.sample((samps_at_a_time, ))
                    norm_samps_tch = torch.as_tensor(norm_samps_tch)
                    sampled_data_lst.append(norm_samps_tch)
                sampled_data = torch.cat(sampled_data_lst, dim=0)[:self.n_aug]
                sampled_data = sampled_data.permute(1, 2, 0, 3).cuda()


            with torch.no_grad():
                sampled_data = sampled_data.reshape(z_support.shape[0], self.n_aug, z_support.shape[2])
                #z_support = torch.tensor(z_support)
                # X_aug = torch.cat([z_support, sampled_data], dim=-2)

            # X_aug_proto = X_aug.contiguous().view(self.n_way, self.n_aug + self.n_aug, -1)
            # z_proto = X_aug_proto.mean(1)#支持集原型是均值
            # z_query = torch.as_tensor(z_query)
        # z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # scores = self.metric(z_query, z_proto)#计算距离

        z_support = z_support.contiguous().view(self.n_way, self.n_support, -1)
        proto = z_support.mean(1).view(self.n_way, -1)
        query = torch.as_tensor(z_query).contiguous().view(self.n_way * self.n_query, -1)
        aug = sampled_data.contiguous().view(self.n_way * self.n_aug, -1)

        return proto, z_support.view(self.n_way * self.n_support, -1), query, aug, int(self.n_aug)

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