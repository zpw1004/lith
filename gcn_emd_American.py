import numpy as np
import torch
from PyEMD import EMD
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity  # Importing cosine_similarity
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
data = pd.read_csv("../dataset/American/train_data.csv")
data = data.dropna()
print(data.shape)
# Extract feature data, excluding the last Facies label
features = data.iloc[:, 3:-1].values

emd = EMD()
imfs_list = [emd.emd(feature).tolist() for feature in features.T]  # Maintain as a list

max_len = max(len(imf) for imfs in imfs_list for imf in imfs)
max_num_imfs = max(len(imfs) for imfs in imfs_list)

padded_imfs_list = []

for imfs in imfs_list:
    # Maintain imfs as list of lists, pad with zero lists where necessary
    imfs += [[0] * max_len] * (max_num_imfs - len(imfs))
    padded_imfs = [imf[:max_len] if len(imf) > max_len else imf + [0] * (max_len - len(imf)) for imf in imfs]
    padded_imfs_list.append(padded_imfs)

imfs_array = np.array(padded_imfs_list).transpose(1, 0, 2)  # Now, convert to a numpy array

similarity_matrix = cosine_similarity(imfs_array.reshape(imfs_array.shape[0], -1))
threshold = 0.1
edges = np.argwhere(similarity_matrix > threshold)
edges = edges[edges[:, 0] != edges[:, 1]]

print(edges)
max_min = preprocessing.StandardScaler()
x = max_min.fit_transform(features)

x = torch.tensor(x, dtype=torch.float)
y = data.iloc[:, -1].values - 1  # 根据您的代码减去1
y = torch.tensor(y, dtype=torch.long)
edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

graph_data = Data(x=x, edge_index=edges, y=y)




test_data = pd.read_csv("../dataset/American/blind_data.csv")
test_data = test_data.dropna()
print(test_data.shape)
# Extract feature data, excluding the last Facies label
test_features = test_data.iloc[:, 3:-1].values

emd = EMD()
imfs_list = [emd.emd(test_feature).tolist() for test_feature in test_features.T]  # Maintain as a list

max_len = max(len(imf) for imfs in imfs_list for imf in imfs)
max_num_imfs = max(len(imfs) for imfs in imfs_list)

padded_imfs_list = []

for imfs in imfs_list:
    # Maintain imfs as list of lists, pad with zero lists where necessary
    imfs += [[0] * max_len] * (max_num_imfs - len(imfs))
    padded_imfs = [imf[:max_len] if len(imf) > max_len else imf + [0] * (max_len - len(imf)) for imf in imfs]
    padded_imfs_list.append(padded_imfs)

imfs_array = np.array(padded_imfs_list).transpose(1, 0, 2)  # Now, convert to a numpy array

similarity_matrix = cosine_similarity(imfs_array.reshape(imfs_array.shape[0], -1))
threshold = 0.1
test_edges = np.argwhere(similarity_matrix > threshold)
test_edges = test_edges[test_edges[:, 0] != test_edges[:, 1]]

print(test_edges)
max_min = preprocessing.StandardScaler()
test_x = max_min.fit_transform(test_features)

test_x = torch.tensor(test_x, dtype=torch.float)
test_y = test_data.iloc[:, -1].values - 1  # 根据您的代码减去1
test_y = torch.tensor(test_y, dtype=torch.long)
test_edges = torch.tensor(test_edges, dtype=torch.long).t().contiguous()

graph_test_data = Data(x=test_x, edge_index=test_edges, y=test_y)










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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

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
    train_accuracy = correct / graph_data.num_nodes

    _, pred = model(graph_test_data).max(dim=1)
    correct = float(pred.eq(graph_test_data.y).sum().item())
    test_accuracy = correct / graph_test_data.num_nodes


    # 打印损失和准确率
    print(f'Epoch [{epoch + 1}/2000], Loss: {loss.item():.4f}, train_Accuracy: {train_accuracy:.4f},test_Accuracy: {test_accuracy:.4f}')

model.eval()
_, pred = model(graph_test_data).max(dim=1)
correct = float(pred.eq(graph_test_data.y).sum().item())
acc = correct /graph_test_data.num_nodes
print('图卷积Accuracy: {:.4f}'.format(acc))