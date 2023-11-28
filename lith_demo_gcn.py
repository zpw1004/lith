import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn import preprocessing
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd

# 加载数据
data = pd.read_csv("../dataset/American/train_data.csv")
data = data.dropna()
# 提取特征和标签
X = data[["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS"]].values
max_min = preprocessing.StandardScaler()
X = max_min.fit_transform(X)
y = data["Facies"].values-1

# 准备数据
# 请确保您已经加载了适当的数据，并将其存储在X和y变量中

# 构建图数据结构
edges = []

# 遍历选择一个要连接到其他所有点的点的索引
for selected_point_index in range(len(X)):
    # 创建连接选定点与其他点之间的连接关系
    edges.extend([(selected_point_index, i) for i in range(len(X)) if i != selected_point_index])


edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
graph = Data(x=torch.tensor(X, dtype=torch.float32), edge_index=edge_index, y=torch.tensor(y, dtype=torch.long))


# 定义一个简单的GCN模型
class RockFaciesClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(RockFaciesClassifier, self).__init__()
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, 16)
        self.line = torch.nn.Linear(16,num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.line(x)
        return x


# 创建模型实例
model = RockFaciesClassifier(num_features=X.shape[1], num_classes=9)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 训练模型
for epoch in range(5000):
    model.train()
    optimizer.zero_grad()
    out = model(graph)
    loss = criterion(out, graph.y)
    loss.backward()
    optimizer.step()
    # 计算准确率
    _, predicted_labels = out.max(dim=1)
    correct = (predicted_labels == graph.y).sum().item()
    accuracy = correct / len(graph.y)

    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')


