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
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import pandas as pd
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from depth.lithology_get_data import GetLoader, get_gcn_newdata

x,y,X_train,y_train,data_trainloader,X_test,y_test = get_gcn_newdata("../output_data/n4_output.csv")




# Create Data objects for training and test sets
train_edges = [[i, j] for i in range(len(X_train)) for j in range(i - 2, i + 3) if i - 2 >= 0 and i + 2 < len(X_train)]
test_edges = [[i, j] for i in range(len(X_test)) for j in range(i - 2, i + 3) if i - 2 >= 0 and i + 2 < len(X_test)]

train_edges = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
test_edges = torch.tensor(test_edges, dtype=torch.long).t().contiguous()

train_graph_data = Data(x=X_train, edge_index=train_edges, y=y_train)
test_graph_data = Data(x=X_test, edge_index=test_edges, y=y_test)


class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = GCNConv(train_graph_data.num_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.gru = nn.GRU(32, 64, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
        self.linear = nn.Linear(128, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        # print(x.shape)
        # 使用GRU处理时间依赖性
        x, _ = self.gru(x)
        # print(x.shape)
        x = x.squeeze(1)  # 取GRU的最后一个时间步作为输出
        # print(x.shape)
        x = self.linear(x)
        return x



# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(9).to(device)
train_graph_data = train_graph_data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=5e-4)

model.train()
for epoch in range(3000):
    optimizer.zero_grad()
    out = model(train_graph_data)
    loss = F.cross_entropy(out, train_graph_data.y)
    loss.backward()
    optimizer.step()
    # 计算准确率
    _, pred = out.max(dim=1)
    correct = pred.eq(train_graph_data.y).sum().item()
    accuracy = correct / train_graph_data.num_nodes

    # 打印损失和准确率
    print(f'Epoch [{epoch + 1}/2000], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

# 预测
model.eval()
_, pred = model(test_graph_data).max(dim=1)
correct = float(pred.eq(test_graph_data.y).sum().item())
acc = correct / test_graph_data.num_nodes
print('图卷积Accuracy: {:.4f}'.format(acc))