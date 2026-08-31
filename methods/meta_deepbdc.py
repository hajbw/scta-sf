import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from .ddwm_utils import Distribution_fitting_with_DDWM, normalize_l2
from .multivariate_laplace import multivariate_laplace
from .template import MetaTemplate
from .bdc_module import BDC

class MetaDeepBDC(MetaTemplate):#父类是metatemplate，对三个抽象方法重写
    def __init__(self, params, model_func, n_way, n_support):
        super(MetaDeepBDC, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()#交叉熵损失
        
        reduce_dim = params.reduce_dim#降维
        self.feat_dim = int(reduce_dim * (reduce_dim+1) / 2)#没懂
        self.dcov = BDC(is_vec=True, input_dim=self.feature.feat_dim, dimension_reduction=reduce_dim)
                   #去BDC生成BDC向量
        self.n_way = n_way
        self.n_support = n_support
    def feature_forward(self, x):#三个抽象方法的重写，特征前向传播
        out = self.dcov(x)#生成BDC向量
        return out

    def set_forward(self, x, is_feature=False):#前向传播，返回距离
        #initial
        z_support, z_query = self.parse_feature(x, is_feature)#返回支持集和查询集特征

        #initial
        #z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1)
        z_proto = z_proto.mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        scores = self.metric(z_query, z_proto)

        '''
        #cross attention: *in dim then sum
        z_support = z_support.reshape(self.n_way, self.n_query, -1, z_support.shape[1])#5,16,1,2064
        #z_proto = z_support.mean(2)
        z_query = z_query.reshape(self.n_way, self.n_query, -1, z_query.shape[1])
        scores = torch.sum(z_support*z_query, dim=-1)#5,16,1
        scores = scores.reshape(self.n_way*self.n_query, -1)
        '''
        '''
        #add gussi sampler for val set
        
        if self.training:
            z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        else:
            n_lsamples = z_support.shape[1]
            n_usamples = z_query.shape[1]
            n_samples = n_lsamples + n_usamples

            # norm：p=2，求L-2范数，L-2规范化
            qry_norm = torch.norm(z_query, p=2, dim=1).unsqueeze(1).expand_as(z_query)
            spt_norm = torch.norm(z_support, p=2, dim=1).unsqueeze(1).expand_as(z_support)
            qry_normalized = z_query.div(qry_norm + 1e-6)
            spt_normalized = z_support.div(spt_norm + 1e-6)
            z_query = qry_normalized.detach()  # 没有梯度
            z_support = spt_normalized.detach()
        '''
        '''
            z_query = qry_normalized.detach().cpu().numpy()  # 没有梯度
            z_support = spt_normalized.detach().cpu().numpy()
        '''
        '''
            #y_support = np.repeat(range(self.n_way), self.n_support)  # 支持集标签

            # ----Transform support sets and query sets with Tukey's Ladder of Power transformation ----#
            #        z_support = torch.pow(z_support, self.dc_tukey_lambda)
            #        z_query = torch.pow(z_query, self.dc_tukey_lambda)
            # ---- distribution calibration and feature sampling
            self.n_aug = 550
            num_sampled = int(self.n_aug / self.n_shot)

            with ((torch.no_grad())):
                mean_tch, cov_tch = Distribution_fitting_with_DDWM(z_support, self.base_means, self.base_means_matrix,
                                                                 self.base_cov, alpha=self.dc_alpha, k=self.dc_k,
                                                               gamma=self.gamma_ddwm)
        '''
        '''
            mean_tch = mean_tch.squeeze()
            cov_tch = cov_tch.squeeze()
        '''
        '''
        
            samps_at_a_time = 1
            with torch.no_grad():
                sampled_data_lst = []
        '''
        '''
                for i in range(mean_tch.shape[0]):
                    mvn_gen = multivariate_laplace(mean_tch[i], cov_tch[i])
                    #mvn_gen = MultivariateNormal(mean_tch, covariance_matrix=cov_tch)
                    for _ in range(int(np.ceil(float(num_sampled) / samps_at_a_time))):
                        #mvn_gen.sample()
                        norm_samps_tch = mvn_gen.rvs(size=samps_at_a_time)
                        #print(_, norm_samps_tch.shape)#1,5,640
                        norm_samps_tch = torch.as_tensor(norm_samps_tch)
                        #norm_samps_tch.shape -> (samps_at_a_time, batch_dim, n_lsamples, n_dim)
                        sampled_data_lst.append(norm_samps_tch)
                sampled_data = torch.stack(sampled_data_lst, dim=0)
                # sampled_data = sampled_data[:num_sampled]
                # sampled_data.shape -> (num_sampled, batch_dim, n_lsamples, n_dim)
                # print(sampled_data.shape)#550,5,640
        '''
                # print(mean_tch.shape, cov_tch.shape)#torch.Size([5, 640]) torch.Size([5, 640, 640])
        '''
                mvn_gen = MultivariateNormal(mean_tch, covariance_matrix=cov_tch)
                for _ in range(int(np.ceil(float(num_sampled) / samps_at_a_time))):
                    mvn_gen.sample()
                    norm_samps_tch = mvn_gen.sample((samps_at_a_time, ))
                    #print(_, norm_samps_tch.shape)#1,5,640
                    norm_samps_tch = torch.as_tensor(norm_samps_tch)
                    #norm_samps_tch.shape -> (samps_at_a_time, batch_dim, n_lsamples, n_dim)
                    sampled_data_lst.append(norm_samps_tch)
                sampled_data = torch.cat(sampled_data_lst, dim=0)[:num_sampled]
                # sampled_data = sampled_data[:num_sampled]
                # sampled_data.shape -> (num_sampled, batch_dim, n_lsamples, n_dim)
                # print(sampled_data.shape)#550,5,640

                sampled_data = sampled_data.permute(1, 2, 0, 3)

                # time_lst_gen.append(time.time() - start_time)

            with torch.no_grad():
                #y_support = torch.tensor(y_support)
                #sampled_label__ = y_support.unsqueeze(-1)
                #sampled_label_ = sampled_label__.expand(n_lsamples, num_sampled)
                #sampled_label = sampled_label_.reshape(n_lsamples * num_sampled)

                sampled_data = sampled_data.reshape(z_support.shape[0], n_lsamples * num_sampled, z_support.shape[2])
                z_support = torch.tensor(z_support)

                #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # z_support = z_support.to(device)
                # y_support = y_support.to(device)
                # sampled_label = sampled_label.to(device)
                # sampled_data = sampled_data.cpu().numpy()
                # print(z_support.is_cuda,sampled_data.is_cuda, 'y', y_support.is_cuda, sampled_label.is_cuda)
                # print(z_support.shape, sampled_data.shape)#torch.Size([5, 640]) torch.Size([2, 2750, 320])
                X_aug = normalize_l2(torch.cat([z_support, sampled_data], dim=-2))
                # X_aug.shape -> batch_dim, n_lsamples + n_lsamples* num_sampled, n_dim
                #Y_aug = torch.cat([y_support, sampled_label], dim=-1)
                # Y_aug.shape -> batch_dim, n_lsamples + n_lsamples*num_sampled

            X_aug_proto = X_aug.contiguous().view(self.n_way, n_lsamples + n_lsamples* num_sampled, -1)
            z_proto = X_aug_proto.mean(1)#支持集原型是均值
            z_query = torch.as_tensor(z_query)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        scores = self.metric(z_query, z_proto)#计算距离
        '''
        return scores

    def set_forward_loss(self, x):#loss前向传播，返回准确率，标签长度，loss，距离

        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.n_way), self.n_query)
        scores = self.set_forward(x)#距离80,1
        if self.training is not True:
            scores = Variable(scores.cuda())


        #initial
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)#选取距离最近的作为预测值
        topk_ind = topk_labels.cpu().numpy()#预测值矩阵
        top1_correct = np.sum(topk_ind[:, 0] == y_label)#预测值和真实值相等就加在准确率上
        '''
        top1_correct = np.sum(preds == y_label)#预测值和真实值相等就加在准确率上
        '''

        #y_query = y_query.cpu()
        correct_this = float(top1_correct)
        count_this = len(y_label)
        loss = self.loss_fn(scores, y_query)

        return correct_this, count_this, loss, scores

    def metric(self, x, y):#xy的BDC距离
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)#断言关键句，真则无行为，假就抛出异常
#squeeze(dim_n)压缩，减少dim_n维度 ，即去掉元素数量为1的dim_n维度。
#unsqueeze(dim_n)，增加dim_n维度，元素数量为1。
        x = x.unsqueeze(1).expand(n, m, d)#变成三维5,2064---5,80,2064
        y = y.unsqueeze(0).expand(n, m, d)#80,2064----5,80,2064

        if self.n_support > 1:#5shot
            dist = torch.pow(x - y, 2).sum(2)#差平方的和
            score = -dist
        else:
            score = (x * y).sum(2)#1shot，矩阵内积就是距离5,80
        return score
