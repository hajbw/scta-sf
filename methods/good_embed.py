import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from .template import MetaTemplate
from sklearn.linear_model import LogisticRegression
from .ddwm_utils import normalize_l2, Distribution_fitting_with_DDWM
from .MvLaplaceSampler import MvLaplaceSampler
from .multivariate_laplace import multivariate_laplace
class GoodEmbed(MetaTemplate):#继承自metatemplate
    def __init__(self, params, model_func, n_way, n_support):#model——func是resnet
        super(GoodEmbed, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()#交叉熵损失
        self.avgpool = nn.AdaptiveAvgPool2d(1)#自适应平均池化
        self.C = params.penalty_C#正则化惩罚因子
        self.params = params

    def feature_forward(self, x):#经过池化层
        out = self.avgpool(x).view(x.size(0),-1)
        return out

    def set_forward(self, x, is_feature=True):#前向传播
        with torch.no_grad():#不进行梯度更新
            z_support, z_query = self.parse_feature(x, is_feature)#分析特征，返回的是支持集和查询集特征
        z_support = z_support.detach()#是返回一个Tensor，它和原张量的数据相同，但requires_grad=False，得到的张量不会具有梯度
        z_query = z_query.detach()#反向传播时调用到detach就会停止

        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        n_lsamples = z_support.shape[0]
        n_usamples = z_query.shape[0]
        n_samples = n_lsamples + n_usamples

        #norm：p=2，求L-2范数，L-2规范化
        qry_norm = torch.norm(z_query, p=2, dim=1).unsqueeze(1).expand_as(z_query)
        spt_norm = torch.norm(z_support, p=2, dim=1).unsqueeze(1).expand_as(z_support)
        qry_normalized = z_query.div(qry_norm + 1e-6)
        spt_normalized = z_support.div(spt_norm + 1e-6)

        z_query = qry_normalized.detach().cpu().numpy()#没有梯度
        z_support = spt_normalized.detach().cpu().numpy()
        y_support = np.repeat(range(self.n_way), self.n_support)#支持集标签

        # ----Transform support sets and query sets with Tukey's Ladder of Power transformation ----#
#        z_support = torch.pow(z_support, self.dc_tukey_lambda)
#        z_query = torch.pow(z_query, self.dc_tukey_lambda)
        # ---- distribution calibration and feature sampling
        self.n_aug = 550
        num_sampled = int(self.n_aug / self.n_shot)

        with torch.no_grad():
            mean_tch, cov_tch = Distribution_fitting_with_DDWM(z_support, self.base_means, self.base_means_matrix,
                                                               self.base_cov, alpha=self.dc_alpha, k=self.dc_k,
                                                               gamma=self.gamma)
        #print('mean_tch, cov_tch is finished!')
        
        samps_at_a_time = 1
        with torch.no_grad():
            sampled_data_lst = []
            #print(mean_tch.shape, cov_tch.shape)#torch.Size([5, 640]) torch.Size([5, 640, 640])
            for i in range(mean_tch.shape[0]):
                mvn_gen = multivariate_laplace(mean_tch[i], cov_tch[i])
#            mvn_gen = MultivariateNormal(mean_tch, covariance_matrix=cov_tch)
                for _ in range(int(np.ceil(float(num_sampled) / samps_at_a_time))):
                #mvn_gen.sample()
                    norm_samps_tch = mvn_gen.rvs(size=samps_at_a_time)
                #print(_, norm_samps_tch.shape)#1,5,640
                    norm_samps_tch = torch.as_tensor(norm_samps_tch)
                # norm_samps_tch.shape -> (samps_at_a_time, batch_dim, n_lsamples, n_dim)
                    sampled_data_lst.append(norm_samps_tch)
            sampled_data = torch.stack(sampled_data_lst, dim=0)
            #sampled_data = sampled_data[:num_sampled]
            # sampled_data.shape -> (num_sampled, batch_dim, n_lsamples, n_dim)
            #print(sampled_data.shape)#550,5,640

            sampled_data = sampled_data.permute(1, 0, 2)

            # time_lst_gen.append(time.time() - start_time)

        with torch.no_grad():
            y_support = torch.tensor(y_support)
            sampled_label__ = y_support.unsqueeze(-1)
            sampled_label_ = sampled_label__.expand(n_lsamples, num_sampled)
            sampled_label = sampled_label_.reshape(n_lsamples * num_sampled)
            sampled_data = sampled_data.reshape(n_lsamples * num_sampled, -1)
            z_support = torch.tensor(z_support)
            
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            #z_support = z_support.to(device)
            #y_support = y_support.to(device)
            #sampled_label = sampled_label.to(device)
            #sampled_data = sampled_data.cpu().numpy()
            #print(z_support.is_cuda,sampled_data.is_cuda, 'y', y_support.is_cuda, sampled_label.is_cuda)
            #print(z_support.shape, sampled_data.shape)#torch.Size([5, 640]) torch.Size([2, 2750, 320])
            X_aug = normalize_l2(torch.cat([z_support, sampled_data], dim=-2))
            # X_aug.shape -> batch_dim, n_lsamples + n_lsamples* num_sampled, n_dim
            Y_aug = torch.cat([y_support, sampled_label], dim=-1)
            # Y_aug.shape -> batch_dim, n_lsamples + n_lsamples*num_sampled


        #逻辑回归，penalty：正则化惩罚项，random_state：随机数种子，C：正则化系数λ的倒数，越小的数值表示越强的正则化。
        #solver：优化算法选择参数，lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。lbfgs需要损失函数的一阶或者二阶连续导数，只能用于L2正则化。
#max_iter：算法收敛最大迭代次数，multi_class：分类方式选择参数，multinomial即前面提到的many-vs-many(MvM)。
#如果模型有T类，我们每次在所有的T类样本里面选择两类样本出来，不妨记为T1类和T2类，把所有的输出为T1和T2的样本放在一起，把T1作为正例，T2作为负例，进行二元逻辑回归，得到模型参数。我们一共需要T(T-1)/2次分类。

        clf = LogisticRegression(penalty='l2',
                                    random_state=0,
                                    C=self.C,
                                    solver='lbfgs',
                                    max_iter=1000,
                                    multi_class='multinomial')
        X_aug = X_aug.cpu().numpy()

        Y_aug = Y_aug.cpu().numpy()

        clf.fit(X_aug, Y_aug)#用支持集拟合分类器
        scores = clf.predict(z_query)#预测查询集标签

        return scores
