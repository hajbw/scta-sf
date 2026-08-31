# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from data.dataset import SetDataset_JSON, SimpleDataset, SetDataset, EpisodicBatchSampler, SimpleDataset_JSON
from abc import abstractmethod


class TransformLoader:
    def __init__(self, image_size):
        self.normalize_param = dict(mean=[0.472, 0.453, 0.410], std=[0.277, 0.268, 0.285])#归一化，RGB三通道
        
        self.image_size = image_size
        print("=====",image_size)
        if image_size == 84:#resnet-12
            self.resize_size = 92#尺寸resize
        elif image_size == 224:#retnet-18
            self.resize_size = 256
        

    def get_composed_transform(self, aug=False):#返回transform
        if aug:#数据增强
            transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),#组成一个compose，随机尺寸裁剪
                transforms.RandomHorizontalFlip(),#随机水平翻转
                transforms.ColorJitter(0.4, 0.4, 0.4),#颜色抖动
                transforms.ToTensor(),#转换为tensor
                transforms.Normalize(**self.normalize_param)#归一化，normalize_param:初始化里面的，规定了mean和std
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.resize_size),#resize，等比缩放
                transforms.CenterCrop(self.image_size),#中心裁剪
                transforms.ToTensor(),#转换为tensor
                transforms.Normalize(**self.normalize_param)#归一化
            ])
        return transform


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass


class SimpleDataManager(DataManager):#pretrain用，distill用
    def __init__(self, data_path, image_size, batch_size, json_read=False):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size#批量数量
        self.data_path = data_path#数据集路径
        self.trans_loader = TransformLoader(image_size)#transform
        self.json_read = json_read

    def get_data_loader(self, data_file, aug):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)#aug=true数据增强
        if self.json_read:
            dataset = SimpleDataset_JSON(self.data_path, data_file, transform)
        else:
            dataset = SimpleDataset(self.data_path, data_file, transform)
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=12, pin_memory=True)#dataloader参数
        #dataset的getitem返回image和target，dataloader的参数：batchsize：一次抓多少牌，shuffle打乱，多进程，加到cuda里
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)#内嵌的方法

        return data_loader


class SetDataManager(DataManager):#剩下的meta、test都用，总之是需要episode的
    def __init__(self, data_path, image_size, n_way, n_support, n_query, n_episode, json_read=False):
        super(SetDataManager, self).__init__()#初始化
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query#batchsize就是支持集+查询集
        self.n_episode = n_episode
        self.data_path = data_path
        self.json_read = json_read

        self.trans_loader = TransformLoader(image_size)#初始化类，

    def get_data_loader(self, data_file, aug):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)#根据aug选择是否数据增强
        if self.json_read:#json
            dataset = SetDataset_JSON(self.data_path, data_file, self.batch_size, transform)#数据集
        else:
            dataset = SetDataset(self.data_path, data_file, self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)#随机抽样
        data_loader_params = dict(batch_sampler=sampler, num_workers=12, pin_memory=True)#batchsize是随机抽样得到的
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)#dataloader
        return data_loader



