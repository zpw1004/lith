import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from sklearn.decomposition import PCA
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
pca = PCA(n_components=2)  # 将特征降至2维
data_pca = pca.fit_transform(X_train)
plt.scatter(data_pca[:, 0], data_pca[:, 1])

# 为每个点添加标签
for i, txt in enumerate(["point1", "point2", "point3", "point4", "point5"]):
    plt.annotate(txt, (data_pca[i, 0], data_pca[i, 1]))

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA 2D Projection')
plt.grid(True)
plt.show()