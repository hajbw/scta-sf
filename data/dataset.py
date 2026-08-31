# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os

identity = lambda x: x

class SimpleDataset:
    def __init__(self, data_path, data_file_list, transform, target_transform=identity):
        label = []
        data = []
        k = 0
        data_dir_list = data_file_list.replace(" ","").split(',')#去掉空格，分割
        for data_file in data_dir_list:
            img_dir = data_path + '/' + data_file
            for i in os.listdir(img_dir):
                file_dir = os.path.join(img_dir, i)#os.path.join:把俩个路径拼接在一起
                for j in os.listdir(file_dir):#os。listdir：把文件夹下面的东西变成列表
                    data.append(file_dir + '/' + j)#数据地址，绝对路径
                    label.append(k)#有多少个
                k += 1
        self.data = data#图片组成的list
        self.label = label
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):#获取每一个图片
        image_path = os.path.join(self.data[i])#每一个图片的位置
        try:
            img = Image.open(image_path).convert('RGB')#转化为RGB，3通道
        except (OSError, IOError) as e:
            print(f"Skipping corrupted file {image_path}: {e}")
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        img = self.transform(img)#modify特征
        target = self.target_transform(self.label[i] - min(self.label))#modify标签
        return img, target

    def __len__(self):
        return len(self.label)#数据集长度


class SetDataset:
    def __init__(self, data_path, data_file_list, batch_size, transform):
        label = []
        data = []
        k = 0
        data_dir_list = data_file_list.replace(" ","").split(',')#去掉空格，逗号分隔
        for data_file in data_dir_list:
            img_dir = data_path + '/' + data_file#图片路径
            for i in os.listdir(img_dir):
                file_dir = os.path.join(img_dir, i)#图片路径+i
                for j in os.listdir(file_dir):
                    data.append(file_dir + '/' + j)#再加j
                    label.append(k)#标签
                k += 1
        self.data = data
        self.label = label
        self.transform = transform

        
        self.cl_list = np.unique(self.label).tolist()
        print(len(self.cl_list))

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.data, self.label):

            self.sub_meta[y].append(x)

        print(len(self.sub_meta))
        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):#转换为tensor类型
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl-0)
        
        return img, target
        

    def __len__(self):
        return len(self.sub_meta)













class SimpleDataset_JSON:
    def __init__(self, data_path, data_file, transform, target_transform=identity):
        data = data_path + '/' + data_file#数据集路径
        with open(data, 'r') as f:
            self.meta = json.load(f)#读取json文件
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = os.path.join(self.meta['image_names'][i])#图片路径
        img = Image.open(image_path).convert('RGB')#转化为三通道
        img = self.transform(img)#modify
        target = self.target_transform(self.meta['image_labels'][i])#标签
        return img, target

    def __len__(self):#长度
        return len(self.meta['image_names'])


class SetDataset_JSON:
    def __init__(self, data_path, data_file, batch_size, transform):
        data = data_path + '/' + data_file#数据路径
        with open(data, 'r') as f:
            self.meta = json.load(f)#json文件

        self.cl_list = np.unique(self.meta['image_labels']).tolist()#转化成列表

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []#没懂

        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            self.sub_meta[y].append(x)#标签加上图片名字

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubDataset_JSON(self.sub_meta[cl], cl, transform=transform)#初始化类
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


class SubDataset_JSON:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        # print( '%d -%d' %(self.cl,i))
        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):#情景批量采样
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes#数据集的类
        self.n_way = n_way#支持集类，也就是要抽取的类
        self.n_episodes = n_episodes#情景

    def __len__(self):#多少个情景
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]#randperm：类随机排列，再抽取n类


