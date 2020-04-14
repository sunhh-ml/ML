# !/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time         : 2020/4/12 18:10
# @Author       : Huanhuan sun
# @Email        : sun58454006@163.com
# @File         : CIFAR10.py
# @Software     : PyCharm
# @Project      : ml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

data_adress = 'E:\\ML_data\\CIFAR10\\'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root=data_adress, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root=data_adress, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# plt.imshow(trainset.data[86])   #   trainset.data中储存了原始数据，并且是array格式，随便看张图

dataiter = iter(trainloader)
images, lables = dataiter.next()
img = torchvision.utils.make_grid(images)
npimg = (img * 0.5 + 0.5).numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))

criteria = nn.CrossEntropyLoss()  # 交叉熵损失
