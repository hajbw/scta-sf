import numpy as np
import torch
from scipy import stats
from torch.distributions import MultivariateNormal
class MvLaplaceSampler:
    def __init__(self, loc, cov):
        '''
        self.mv_normal_list = []
        self.normal_list = []
        self.laplace_list = []
        '''
        self.loc = loc
        self.cov = cov
        self.var = torch.diagonal(cov, dim1=-1, dim2=-2)
        self.mv_normal = stats.multivariate_normal(self.loc, cov=self.cov)
        self.normal = stats.norm(loc=self.loc, scale=np.sqrt(self.var))
        self.laplace = stats.laplace(loc=self.loc, scale=np.sqrt(self.var / 2))
        '''
        for i in range(loc.shape[0]):
            self.var_i = np.diag(cov[i])

            self.mv_normal_i = MultivariateNormal(self.loc[i], cov=self.cov[i], allow_singular=True)
            self.normal_i = stats.norm(loc=self.loc[i], scale=np.sqrt(self.var_i))
            self.laplace_i = stats.laplace(loc=self.loc[i], scale=np.sqrt(self.var_i/2))
            self.mv_normal_list.append(self.mv_normal_i)
            self.normal_list.append(self.normal_i)
            self.laplace_list.append(self.laplace_i)
         '''
        
    def sample(self, sample_size: int=None):
        #self.mv_normal.sample()
        mv_samples = self.mv_normal.rvs(sample_size)
        cdf_samples = self.normal.cdf(mv_samples)
        laplace_samples = self.laplace.ppf(cdf_samples)
        '''
        mv_samples = []
        cdf_samples = []
        laplace_samples = []
        for i in range(self.loc.shape[0]):
            #print(self.mv_normal_list[i].dtype)
            mv_samples.append(self.mv_normal_list[i].rvs(sample_size))
            cdf_samples.append(self.normal_list[i].cdf(mv_samples[i]))
            laplace_samples_i = self.laplace_list[i].ppf(cdf_samples[i])
            laplace_samples_i = torch.tensor(laplace_samples_i)
            laplace_samples.append(laplace_samples_i)
            
        laplace_samples = torch.stack(laplace_samples, 0)
        '''
        return laplace_samples
