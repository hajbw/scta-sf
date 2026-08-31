import numpy as np
import torch

def shorten_long_axis(cov: torch.Tensor, mode: str = 'match_second', factor: float = 0.5):
    """
    批量处理 B*N*N 协方差矩阵，压制长轴或进行球化。

    Args:
        cov (torch.Tensor): 形状 (B, N, N) 的批量协方差矩阵（必须正定）。
        mode (str): 
            - 'match_second': 最大特征值砍到第二大（剧烈变圆，总方差减小）。
            - 'shrink_first': 最大特征值向第二大插值（平滑收缩长轴）。
            - 'sphere_smooth': **【新增】平滑球化，0=完全不变，1=完全球化（总方差不变）**。
            - 'spherify': 强制所有特征值等于自身均值（完全球化，总方差不变）。
        factor (float): 
            - 'shrink_first' 时：0=砍到第二大，1=不变。
            - 'sphere_smooth' 时：**0=不变（原始），1=完全球化**。

    Returns:
        torch.Tensor: 形状 (B, N, N) 修改后的批量协方差矩阵。
    """
    B, N = cov.shape[0], cov.shape[1]
    # print(cov.shape)
    
    # 边界情况：1维分布不存在“形状”概念，直接返回
    if N == 1:
        return cov.clone()

    # 1. 批量特征分解（eigh 返回升序小->大，翻转后变为大->小）
    eigvals_asc, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals_asc.flip(-1)          # (B, N)
    eigvecs = eigvecs.flip(-1)              # (B, N, N)

    new_vals = eigvals.clone()
    # print("before:",eigvals[:,:10],eigvals[:,-10:])

    # 2. 根据模式修改特征值
    if mode == 'match_second':
        new_vals[:, 0] = eigvals[:, 1]
        
    elif mode == 'shrink_first':
        if not (0 <= factor <= 1):
            raise ValueError("factor 必须在 [0, 1] 区间内")
        new_vals[:, 0] = eigvals[:, 0] * factor + eigvals[:, 1] * (1 - factor)
        
    elif mode == 'sphere_smooth':
        # 【新增模式】因子：0=原始(不变)，1=完全球化
        if not (0 <= factor <= 1):
            raise ValueError("factor 必须在 [0, 1] 区间内")
        # 计算每个批次样本的特征值均值 (B, 1)
        mean_vals = eigvals.mean(dim=-1, keepdim=True)
        # 向均值线性插值：factor=1 时完全变为均值（球化），factor=0 时保留原样
        new_vals = eigvals * (1 - factor) + mean_vals * factor
        
    elif mode == 'spherify':
        # 极端情况：直接全部赋值为均值（等同于 sphere_smooth 的 factor=1）
        mean_vals = eigvals.mean(dim=-1, keepdim=True)
        # new_vals = mean_vals.expand(-1, N)
        new_vals = mean_vals.expand(-1, N)
    else:
        raise ValueError("未知 mode，请选择 'match_second', 'shrink_first', 'sphere_smooth' 或 'spherify'")
    
    # print("after:",new_vals[:,:10],new_vals[:,-10:])

    # 3. 重构协方差矩阵
    diag_new_vals = torch.diag_embed(new_vals)
    cov_new = eigvecs @ diag_new_vals @ eigvecs.transpose(-2, -1)
    
    # 4. 强制对称化（消除浮点误差）
    cov_new = (cov_new + cov_new.transpose(-2, -1)) / 2
    
    return cov_new

def normalize_l2(x, dim=-1):
    '''x.shape = (batch_dim, n_lsamples + n_lsamples* num_sampled, n_dim)'''
    x_norm = torch.linalg.norm(x, dim=dim, keepdims=True)
    x = torch.div(x, x_norm)
    return x
def Distribution_fitting_with_DDWM(query, base_means, base_means_matrix, base_cov, k, alpha):
    #print(type(query))
    #query = torch.from_numpy(query)
    '''
    assert torch.is_tensor(query)
    if query.shape[1] > 1:
        query = query.mean(1).reshape(query.shape[0], 1, query.shape[2])
    else:
        query = query
    '''
    base_means = normalize_l2(base_means)

    #print(query.shape)#5,640
    assert torch.is_tensor(base_means)#5,64
    assert torch.is_tensor(base_cov)#5,640,640

    batch_dims, n_dim = query.shape[:-1], query.shape[-1]#5,640
    batch_dim = int(np.prod(batch_dims))#5

    n_classes = base_means.shape[0]#64
    assert base_means.shape == (n_classes, n_dim)
    assert base_cov.shape == (n_classes, n_dim, n_dim)

    base_means = base_means.unsqueeze(0).expand(batch_dim, n_classes, n_dim)
    base_cov = base_cov.unsqueeze(0).expand(batch_dim, n_classes, n_dim, n_dim)
    #print(base_means.shape)#5,64,640
    #assert query.shape == (batch_dim, n_dim)
    #print('meiyoucuowu')
    # query      -> shape = (batch_dim, n_dim)
    # base_means -> shape = (batch_dim, n_classes, n_dim)
    # base_cov   -> shape = (batch_dim, n_classes, n_dim, n_dim)
    # --- Calculate the feature description matrix of support samples --- #
    query_matrix = torch.matmul(query.reshape(batch_dim, 1, n_dim, 1),
                                query.reshape(batch_dim, 1, n_dim, 1).permute(0, 1, 3, 2))
    #print(query_matrix.shape,base_means_matrix.shape)#5,1,640,640  [25, 64, 640, 640]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #query_matrix = query_matrix.to(device)
    base_means = base_means.to(device)
    base_cov = base_cov.to(device)
    #print(query_matrix.is_cuda,base_means_matrix.is_cuda)
    # --- Calculate Frobenius norm values of the difference and Select k nearest base classes--- #
    matrix_L2_dist =  torch.linalg.norm(query_matrix - base_means_matrix, ord='fro', dim=(2, 3))
    #print(matrix_L2_dist.shape)#torch.Size([5, 64])
    index = torch.topk(matrix_L2_dist, k, dim=-1, largest=False, sorted=True).indices  # index.shape == (batch_dim, k)

    # --- Calculate weight factors of k nearest base classes --- #
    #print(base_means.is_cuda)
    #base_means = base_means.to(device)
    #query = query.to(device)

    dist = torch.linalg.norm(query.reshape(batch_dim, 1, n_dim) - base_means, 2,
                             dim=-1)  # dist.shape == (batch_dim, n_classes)
    Weight = torch.div(1, dist)
    #Weight = torch.div(1, torch.pow(1 + dist, gamma))
    '''
    query_matrix = query_matrix.cpu()
    matrix_L2_dist = matrix_L2_dist.cpu()
    torch.cuda.empty_cache()
    '''

    gather_weight = torch.gather(Weight, dim=-1, index=index).unsqueeze(-1).reshape(batch_dim, k, 1)
    assert gather_weight.shape == (batch_dim, k, 1)
    #base_means=base_means
    #index=index.to(device)
    #base_cov=base_cov.to(device)

    # --- Calculate the weighted mean and Covariance of base classes --- #
    gathered_mean = torch.gather(base_means, dim=-2, index=index.unsqueeze(-1).expand(batch_dim, k, n_dim))
    assert gathered_mean.shape == (batch_dim, k, n_dim)
    gathered_cov = torch.gather(base_cov, dim=-3, index=index.unsqueeze(-1).unsqueeze(-1).expand(batch_dim, k, n_dim, n_dim))
    assert gathered_cov.shape == (batch_dim, k, n_dim, n_dim)

    Weight_gathered_mean = torch.matmul(gathered_mean.permute(0, 2, 1), gather_weight).reshape(batch_dim, n_dim)
    assert Weight_gathered_mean.shape == ((batch_dim, n_dim))
    Weight_gathered_cov = torch.sum(gathered_cov * gather_weight.reshape(batch_dim, k, 1, 1), dim=1)
    assert Weight_gathered_cov.shape == (batch_dim, n_dim, n_dim)
    '''
    gathered_mean = gathered_mean.cpu()
    gathered_cov = gathered_cov.cpu()
    torch.cuda.empty_cache()
    '''
    # --- Calculate the mean and the covariance of the learned feature distribution --- #
    #Weight_gathered_mean=Weight_gathered_mean.cpu()
    #query=query.cpu()
    learned_mean = torch.div((Weight_gathered_mean + query.reshape(batch_dim, n_dim)), torch.sum(gather_weight, dim=1) + 1)
    assert learned_mean.shape == (batch_dim, n_dim) # learned_mean.shape == (batch_dim, n_dim)

    learned_cov = torch.div(Weight_gathered_cov + alpha, torch.sum(gather_weight, dim=1).reshape(batch_dim, 1, 1) + 1)
    #print(learned_cov.dtype)#64
    learned_cov = learned_cov.to(device)
    learned_cov = learned_cov + 1e-6 * torch.eye(n_dim).unsqueeze(0).expand(batch_dim, n_dim, n_dim).to(
        device, dtype=torch.float32)
    assert learned_cov.shape == (batch_dim, n_dim, n_dim)  # learned_cov.shape == (batch_dim, n_dim, n_dim)
    '''
    Weight_gathered_mean = Weight_gathered_mean.cpu()
    Weight_gathered_cov = Weight_gathered_cov.cpu()
    gather_weight = gather_weight.cpu()
    torch.cuda.empty_cache()
    '''
    
    # print(f"learned_mean.shape: {learned_mean.shape}, learned_cov.shape: {learned_cov.shape}")
    learned_cov = shorten_long_axis(learned_cov, mode='sphere_smooth', factor=0.8)
    mean_tch = learned_mean.reshape(*batch_dims, n_dim)
    cov_tch = learned_cov.reshape(*batch_dims, n_dim, n_dim)
    cov_tch = cov_tch.to(torch.float32)

    # cov_tch = shorten_long_axis(cov_tch, mode='spherify', factor=0.9)


    # cov_tch *= 0.25

    return mean_tch, cov_tch
