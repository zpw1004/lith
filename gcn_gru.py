import torch
import torch.nn.functional as F
from sklearn import preprocessing
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import pandas as pd
import torch.nn as nn
# # 加载数据
# data = pd.read_csv("../dataset/American/train_data.csv")
# # data_test = pd.read_csv("../dataset/American/blind_data.csv")
# data = data.dropna()
# # data_test = data_test.dropna()
#
# # 创建图的边
# # edges = [[i, i+1] for i in range(len(data)-1)] + [[i+1, i] for i in range(len(data)-1)]
# # edges = [[i, i + 2] for i in range(len(data) - 2)] + [[i + 2, i] for i in range(len(data) - 2)]
# edges = [[i, j] for i in range(len(data)) for j in range(i - 2, i + 3) if i - 2 >= 0 and i + 2 < len(data)]
# # edges = [[i, j] for i in range(len(data)) for j in range(i - 2, i + 3) if i - 2 >= 0 and i + 2 < len(data) and i != j]
# print(edges)
# edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
# # 创建PyTorch Geometric的Data对象
#
#
# # 从表格中提取特征和标签
# # 注意：您需要使用data.values来从DataFrame中获取值，而不是直接使用data
# x = data.iloc[:, 2:-1].values
# max_min = preprocessing.StandardScaler()
# x = max_min.fit_transform(x)
# x = torch.tensor(x, dtype=torch.float)
#
# # 假设最后一列是岩性标签
# y = data.iloc[:, -1].values - 1  # 根据您的代码减去1
# y = torch.tensor(y, dtype=torch.long)
# # 创建PyTorch Geometric的Data对象
# # print(x.shape)
# # x = x.unsqueeze(0)  # 添加批量维度
# # print(x.shape)
# # 创建PyTorch Geometric的Data对象
# graph_data = Data(x=x, edge_index=edges, y=y)
from depth.lithology_get_data import get_American_data

graph_data,graph_test_data = get_American_data("../dataset/American/train_data.csv","../dataset/American/blind_data.csv")


class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = GCNConv(graph_data.num_features, 16)
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
graph_data = graph_data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=5e-4)

model.train()
for epoch in range(400):
    optimizer.zero_grad()
    out = model(graph_data)
    # print(out.shape)
    # print(graph_data.y.shape)
    loss = F.cross_entropy(out, graph_data.y)
    loss.backward()
    optimizer.step()
    # 计算准确率
    _, pred = out.max(dim=1)
    correct = pred.eq(graph_data.y).sum().item()
    accuracy = correct / graph_data.num_nodes

    # 打印损失和准确率
    print(f'Epoch [{epoch + 1}/2000], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

# 预测
model.eval()
_, pred = model(graph_test_data).max(dim=1)
correct = float(pred.eq(graph_test_data.y).sum().item())
acc = correct /graph_test_data.num_nodes
print('图卷积Accuracy: {:.4f}'.format(acc))










#
# print("对比+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#
# class Net_Line(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lin1 = nn.Linear(8, 128)
#         self.lin2 = nn.Linear(128, 9)
#
#     def forward(self, x):
#         x = F.leaky_relu(self.lin1(x))
#         x = F.leaky_relu(self.lin2(x))
#         return x
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net_Line().to(device)
# x = x.to(device)
# y = y.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=5e-4)
#
# model.train()
# for epoch in range(6000):
#     optimizer.zero_grad()
#     out = model(x)
#     loss = F.cross_entropy(out,y)
#     loss.backward()
#     optimizer.step()
#     # 计算准确率
#     _, pred = out.max(dim=1)
#     correct = pred.eq(y).sum().item()
#     accuracy = correct / y.size(0)
#
#     # 打印损失和准确率
#     print(f'Epoch [{epoch + 1}/2000], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
#
# # 预测
# model.eval()
# _, pred = model(x).max(dim=1)
# correct = float(pred.eq(y).sum().item())
# acc = correct / y.size(0)
# print('线性层Accuracy: {:.4f}'.format(acc))