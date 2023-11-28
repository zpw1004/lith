import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils import data
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter
from sklearn import preprocessing
from sklearn import model_selection
import matplotlib.pyplot as plt

from depth.lithology_get_data import GetLoader,get_saling_data

X_train,y_train,data_trainloader,X_test,y_test = get_saling_data("output_data/n4_output.csv")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(11, 128)
        self.lin2 = nn.Linear(128, 256)
        self.lin3 = nn.Linear(256, 4)
        self.bn_in = nn.BatchNorm1d(11)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x_in):
        # print(x_in.shape)
        x = self.bn_in(x_in)
        x = F.leaky_relu(self.lin1(x))
        x = self.bn1(x)
        # print(x)

        x = F.leaky_relu(self.lin2(x))
        x = self.bn2(x)
        # print(x)

        x = F.leaky_relu(self.lin3(x))
        return x

#实例化模型
model = Net()
#学习率
Learning_rate = 0.01
#优化器
optimizer = torch.optim.Adam(model.parameters(),lr=Learning_rate)
#optimizer = torch.optim.SGD(model.parameters(),lr=Learning_rate,momentum=0.9,weight_decay=1e-4)
#损失函数
criterion = nn.CrossEntropyLoss()

model.train()
Total_epoch = 300
best = 0
# 记录损失函数：
losses = []
for epoch in range(Total_epoch):
    for i, (x, y) in enumerate(data_trainloader):
        optimizer.zero_grad()
        outputs = model(x)
        # 计算损失函数
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().data.item())
    if epoch % 100 == 0:
        print('epoch: %d/%d, Loss: %.4f' % (epoch, Total_epoch, np.mean(losses)))

    # 测试准确率
    correct = 0
    total = 0
    x = torch.FloatTensor(X_test)
    y = torch.LongTensor(y_test)
    outputs = model(x)
    _, predicted = torch.max(outputs.data, 1)
    total += y.size(0)
    correct += (predicted == y).sum()
    acc = 100 * correct / total
    print('epoch: %d/%d,准确率：%.4f %%' % (epoch, Total_epoch, acc))
    if acc > best:
        best = acc
        # 保存模型 torch.save(state,path)
        path = './linear_model_best.pth'
        checkpoint = torch.save(model.state_dict(), path)

model_load = Net()
model_load.load_state_dict(torch.load(path),False)

correct = 0
total = 0
x = torch.FloatTensor(X_test)
y = torch.LongTensor(y_test)
outputs = model_load(x)
_,predicted = torch.max(outputs.data,1)
total += y.size(0)
correct += (predicted ==y).sum()
print('准确率：%.4f %%'%(100*correct/total))