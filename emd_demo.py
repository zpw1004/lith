import numpy as np
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops
# 输入数据
data_np = np.array([
    [2793, 77.45, 0.664, 9.9, 11.915, 4.6, 1, 1, 3],
    [2793.5, 78.26, 0.661, 14.2, 12.565, 4.1, 1, 0.979, 3],
    [2794, 79.05, 0.658, 14.8, 13.05, 3.6, 1, 0.957, 3],
    [2794.5, 86.1, 0.655, 13.9, 13.115, 3.5, 1, 0.936, 3],
    [2795, 74.58, 0.647, 13.5, 13.3, 3.4, 1, 0.915, 3],
    [2795.5, 73.97, 0.636, 14, 13.385, 3.6, 1, 0.894, 3],
    [2796, 73.72, 0.63, 15.6, 13.93, 3.7, 1, 0.872, 3],
    [2796.5, 75.65, 0.625, 16.5, 13.92, 3.5, 1, 0.83, 3],
    [2797, 73.79, 0.624, 16.2, 13.98, 3.4, 1, 0.809, 3],
    [2797.5, 76.89, 0.615, 16.9, 14.22, 3.5, 1, 0.787, 3],
    [2798, 76.11, 0.6, 14.8, 13.375, 3.6, 1, 0.766, 3],
    [2798.5, 74.95, 0.583, 13.3, 12.69, 3.7, 1, 0.745, 3],
    [2799, 71.87, 0.561, 11.3, 12.475, 3.5, 1, 0.723, 3],
    [2799.5, 83.42, 0.537, 13.3, 14.93, 3.4, 1, 0.702, 3],
    [2800, 90.1, 0.519, 14.3, 16.555, 3.2, 1, 0.681, 2],
    [2800.5, 78.15, 0.467, 11.8, 15.96, 3.1, 1, 0.638, 2],
    [2801, 69.3, 0.438, 9.5, 15.12, 3.1, 1, 0.617, 2],
    [2801.5, 63.54, 0.418, 8.8, 15.19, 3, 1, 0.596, 2],
    [2802, 63.87, 0.401, 7.2, 15.39, 2.9, 1, 0.574, 2]
])

# Extract depth, features, and labels from the NumPy array
depth = data_np[:, 0]
features = data_np[:, 1:-1]
labels = data_np[:, -1]

# Create edges dynamically based on depth proximity
edges = []
for i in range(len(depth)):
    for j in range(i + 1, len(depth)):
        if abs(depth[i] - depth[j]) <= 0.5:
            edges.append((i, j))

# Define the Graph Data
edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
x = torch.tensor(features, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.long)

# Create a Graph Data object
data = Data(x=x, edge_index=edge_index, y=y)


# Define a Graph Convolutional Network (GCN) model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize the model, optimizer, and loss function
model = GCN(input_dim=features.shape[1], hidden_dim=64, num_classes=int(max(labels)) + 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training loop (you can customize the number of epochs)
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Set training and test masks (you can customize this)
data.train_mask = torch.tensor([True] * 12 + [False] * (len(depth) - 12), dtype=torch.bool)
data.test_mask = ~data.train_mask

# Training the model
for epoch in range(100):
    loss = train()
    print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')

# Evaluate the model on the test set
model.eval()
pred = model(data)[data.test_mask].max(1)[1]
correct = pred.eq(data.y[data.test_mask]).sum().item()
total = data.test_mask.sum().item()
accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')