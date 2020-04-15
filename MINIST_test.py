# !/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time         : 2020/4/15 12:43
# @Author       : Huanhuan sun
# @Email        : sun58454006@163.com
# @File         : MINIST_test.py
# @Software     : PyCharm
# @Project      : ml

from pathlib import Path
import requests
import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from IPython.core.debugger import set_trace
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


DATA_PATH = Path("e:\\ML_data\\MINIST\\")
PATH = DATA_PATH / "minist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# plt.imshow(x_train[0].reshape((28, -1)), cmap="gray")
# print(x_train.shape)

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
n, c = x_train.shape
# x_train, x_train.shape, y_train.min(), y_train.max()
# print(x_train, y_train)
# print(x_train.shape)
# print(y_train.min, y_train.max)



class mnist(nn.Module):
    def __init__(self):
        super(mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 12, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # self.conv3 = nn.Conv2d(12, 20, 3, padding=1)
        # self.conv4 = nn.Conv2d(20, 28, 3, padding=1)
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(2, 2)
        #
        # self.conv5 = nn.Conv2d(28, 28, 3, padding=1)
        # self.conv6 = nn.Conv2d(28, 56, 3)
        # self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(12 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.relu2(x)
        # x = self.pool2(x)
        #
        # x = self.conv5(x)
        # x = self.conv6(x)
        # x = self.relu3(x)
        print(x.size())
        x = x.view(-1, 56 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


epochs = 2
batch_size = 64
model = mnist()
optimzier = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
critiria = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for i in range((n - 1) // batch_size + 1):
        start_i = i * batch_size
        end_i = start_i + batch_size
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = critiria(pred, yb)

        loss.backward()
        optimzier.step()
        optimzier.zero_grad()

print(critiria(model(xb), yb))






















# loss_fn = F.cross_entropy
#
#
# class Mnist_Logistic(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
#         self.bias = nn.Parameter(torch.zeros(10))
#
#     def forward(self, xb):
#         return xb @ self.weights + self.bias
#
#
# model = Mnist_Logistic()
# bs = 64
# xb = x_train[0:bs]
# preds = model(xb)
# print(preds[0], preds.shape)
#
# yb = y_train[0:bs]
# print(loss_fn(preds, yb))
#
#
# def accuracy(out, yb):
#     preds = torch.argmax(out, dim=1)
#     return (preds == yb).float().mean()
#
#
# print(accuracy(preds, yb))
# lr = 0.5
# epochs = 60
#
# # def fit():
# for epoch in range(epochs):
#     for i in range((n - 1) // bs + 1):
#         # set_trace()       #  有这句时，每次循环到这里会暂停，可以输入p 变量来查看，输入c可以继续循环，exit可以暴力退出整个程序终止
#         start_i = i * bs
#         end_i = start_i + bs
#         xb = x_train[start_i:end_i]
#         yb = y_train[start_i:end_i]
#         pred = model(xb)
#         loss = loss_fn(pred, yb)
#
#         loss.backward()
#         with torch.no_grad():
#             for p in model.parameters():
#                 p -= p.grad * lr
#             model.zero_grad()
#     # print('=============================================')
#     # print("epoch = ", epoch)
#
# # fit()
# print(loss_fn(model(xb), yb), accuracy(model(xb), yb))
