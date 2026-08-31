import collections
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import time
import os
from methods.stl_deepbdc import STLDeepBDC

from methods.good_embed import GoodEmbed
from data.datamgr import SetDataManager, SimpleDataManager

from methods.meta_deepbdc_dwdwm import MetaDeepBDC
from utils import *
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', default=84, type=int, choices=[84, 224])
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--dataset', default='mini_imagenet', choices=['mini_imagenet', 'archive','tiered_imagenet', 'cub', 'cifar', 'fc100'])
parser.add_argument('--data_path', type=str)
parser.add_argument('--model', default='ResNet12', choices=['ResNet12', 'ResNet18'])
parser.add_argument('--method', default='stl_deepbdc', choices=['meta_deepbdc', 'stl_deepbdc', 'protonet', 'good_embed', 'protonet_noattn', 'protonet_aug'])

parser.add_argument('--test_n_way', default=5, type=int, help='number of classes used for testing (validation)')
parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
parser.add_argument('--n_query', default=15, type=int, help='number of unlabeled data in each class during meta validation')

parser.add_argument('--test_n_episode', default=2000, type=int, help='number of episodes in test')
parser.add_argument('--model_path', default='', help='meta-trained or pre-trained model .tar file path')
parser.add_argument('--test_task_nums', default=5, type=int, help='test numbers')#测试数量
parser.add_argument('--gpu', default='0', help='gpu id')
parser.add_argument('--distance', default='cosine', choices=['cosine', 'euclidean', 'inner_product'], help='distance function for protoNet')
parser.add_argument('--penalty_C', default=0.1, type=float, help='logistic regression penalty parameter')#逻辑回归的惩罚参数
parser.add_argument('--reduce_dim', default=64, type=int, help='the output dimensions of BDC dimensionality reduction layer')#降维层
parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate for pretrain and distillation')#在训练过程中，随机把一些 input（输入的tensor数据类型）中的一些元素变为0，变为0的概率为p

parser.add_argument('--dc_tukey_lambda', default=0.5, type=float, help='')
#parser.add_argument('--gamma', default=0.5, type=float, help='')
parser.add_argument('--n_aug', default=5000, type=int, help='')
parser.add_argument('--dc_k', default=4, type=int, help='')
parser.add_argument('--dc_alpha', default=0.05, type=float, help='')

task_bs = 1  # The number of tasks to stack to each other for parallel optimization

params = parser.parse_args()
num_gpu = set_gpu(params)#设置GPU


json_file_read = False
if params.dataset == 'mini_imagenet':  # 不同数据集的参数设置
    base_file = 'train'
    novel_file = 'test'  # 测试集
    params.num_classes = 64
elif params.dataset == 'cub':
    base_file = 'base.json'
    val_file = 'val.json'
    novel_file = 'novel.json'  # 新类
    json_file_read = True
    params.num_classes = 200
elif params.dataset == 'tiered_imagenet':
    base_file = 'train'
    novel_file = 'test'  # 测试集
    params.num_classes = 351
elif params.dataset == 'cifar':
    base_file = 'train'
    novel_file = 'test'
    params.num_classes = 64
elif params.dataset == 'fc100':
    base_file = 'train'
    novel_file = 'test'
    params.num_classes = 60
elif params.dataset == 'archive':
    base_file = 'train'
    novel_file = 'test'
    params.num_classes = 10
else:
    ValueError('dataset error')
# 实例化对象simpledatamanager，调用方法
base_datamgr = SimpleDataManager(params.data_path, params.image_size, batch_size=params.batch_size,
                                 json_read=json_file_read)
base_loader = base_datamgr.get_data_loader(base_file, aug=True)  # 数据增强

novel_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)#新类的参数
novel_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.test_n_episode, json_read=json_file_read,  **novel_few_shot_params)
novel_loader = novel_datamgr.get_data_loader(novel_file, aug=False)#情景训练，不进行数据增强


if params.method == 'meta_deepbdc':
    model = MetaDeepBDC(params, model_dict[params.model], **novel_few_shot_params)
elif params.method == 'protonet':#选择方法
    from methods.protonet import ProtoNet
    model = ProtoNet(params, model_dict[params.model], **novel_few_shot_params)
elif params.method == 'protonet_aug':#选择方法
    from methods.protonet_dwdwm import ProtoNet
    model = ProtoNet(params, model_dict[params.model], **novel_few_shot_params)
elif params.method == 'protonet_noattn':#选择方法
    from methods.protonet_dwdwm import ProtoNet_noattn
    model = ProtoNet_noattn(params, model_dict[params.model], **novel_few_shot_params)
elif params.method == 'good_embed':
    model = GoodEmbed(params, model_dict[params.model], **novel_few_shot_params)
elif params.method == 'stl_deepbdc':
    model = STLDeepBDC(params, model_dict[params.model], **novel_few_shot_params)

# model save path
model = model.cuda()
model.eval()#验证

print(params.model_path)
model_file = os.path.join(params.model_path)
model = load_model(model, model_file)#加载模型

# ---- Base class statistics
base_stats_path = f'./basestats/{params.dataset}.pth'
os.makedirs('./basestats', exist_ok=True)

if os.path.exists(base_stats_path):
    print(f"Loading base stats from {base_stats_path}")
    stats = torch.load(base_stats_path)
    base_means = stats['base_means'].cuda()
    base_cov = stats['base_cov'].cuda()
    base_means_matrix = stats['base_means_matrix'].cuda()
else:
    print("Calculating base stats...")
    base_means = []
    base_cov = []

    #model1 = wrn_model.wrn28_10(num_classes=params.num_classes)

    with torch.no_grad():
        output_dict = collections.defaultdict(list)

        for i, (inputs, labels) in enumerate(base_loader):
            # compute output
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model.feature.forward(inputs)#resnet 64,640,10,10
            outputs = model.feature_forward(outputs)#pooling
    #        print(outputs.shape)#stl:64,8256,avgpool:64,640
            outputs = outputs.cpu().data.numpy()
            for out, label in zip(outputs, labels):
                output_dict[label.item()].append(out)

        all_info = output_dict
        for key in all_info.keys():
            feature = np.array(all_info[key])
            mean = np.mean(feature, axis=0)
            cov = np.cov(feature.T)
            base_means.append(mean)
            base_cov.append(cov)
    print("* Means and Covariance Matrices are calculated")
    #print(params.batch_size)
    # --- Calculate feature description matrices of each base class ---#

    with torch.no_grad():
        base_means = torch.cat(
            [torch.from_numpy(x).float().unsqueeze(0)
             for x in base_means])
        base_cov = torch.cat(
            [torch.from_numpy(x).float().unsqueeze(0)
             for x in base_cov])
        #print(base_means.shape, base_means.device)#64,8256 cpu
        base_means_matrix = torch.matmul(
            base_means.unsqueeze(-1).expand(task_bs * params.test_n_way * params.n_shot, base_means.shape[0], base_means.shape[1], 1),
            base_means.unsqueeze(-1).expand(task_bs * params.test_n_way * params.n_shot, base_means.shape[0], base_means.shape[1], 1).permute(0, 1, 3, 2))
        #print(base_means_matrix.size, base_means_matrix.device)
        base_means = base_means.cuda()
        base_cov = base_cov.cuda()
        base_means_matrix = base_means_matrix.cuda()
    # Save the stats
    stats = {
        'base_means': base_means,
        'base_cov': base_cov,
        'base_means_matrix': base_means_matrix
    }
    torch.save(stats, base_stats_path)
    print(f"Saved base stats to {base_stats_path}")


print(params)
iter_num = params.test_n_episode#测试的episode，2000
acc_all_task = []#所有test的准确率
for _ in range(params.test_task_nums):#5轮
    acc_all = []
    test_start_time = time.time()#开始时间
    tqdm_gen = tqdm.tqdm(novel_loader)#进度条的长度
    for _, run_idxs in enumerate(tqdm_gen):
        with torch.no_grad():#不进行梯度更新
            model.batch_dim = len(run_idxs)
            (x, _) = run_idxs

            model.n_query = params.n_query#15
            model.base_means = base_means
            model.base_means_matrix = base_means_matrix
            model.base_cov = base_cov
            model.dc_alpha = params.dc_alpha
            model.dc_k = params.dc_k
            model.n_aug = params.n_aug
            model.dc_tukey_lambda = params.dc_tukey_lambda
            #model.gamma_ddwm = params.gamma
            model.dc_tukey_lambda = params.dc_tukey_lambda
            model.n_shot = params.n_shot

            scores = model.set_forward(x, False)#模型前向传播得到预测
        if params.method in ['meta_deepbdc', 'protonet']:#scores是距离
            pred = scores.data.cpu().numpy().argmax(axis=1)#argmax：选出每一行的最大值的位置
        else:
            pred = scores#goodembed和stl的输出是预测值

        y = np.repeat(range(params.test_n_way), params.n_query)
        acc = np.mean(pred == y) * 100#对应位置相等，加起来取mean

        acc_all.append(acc)
        tqdm_gen.set_description(f'avg.acc:{(np.mean(acc_all)):.2f} (curr:{acc:.2f})')#平均acc和现在的acc

    acc_all = np.asarray(acc_all)#作为矩阵
    acc_mean = np.mean(acc_all)#求平均
    acc_std = np.std(acc_all)#求方差
    #打印准确率的置信区间
    print('%d Test Acc = %4.2f%% +- %4.2f%% (Time uses %.2f minutes)'
        % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num), (time.time() - test_start_time) / 60))
    acc_all_task.append(acc_all)

acc_all_task_mean = np.mean(acc_all_task)#5轮的平均准确率，共10000个task
print('%d test mean acc = %4.2f%%' % (params.test_task_nums, acc_all_task_mean))


