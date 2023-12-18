import argparse
import pickle
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='daqing', help='Dataset to use: daqing or xinjiang')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=244, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR scheduler gamma.')
    parser.add_argument('--step-size', type=int, default=20, help='LR scheduler step size.')
    parser.add_argument('--data_path1', type=str, default="dataset/example_daqing.xlsx", help='daqing_path')
    parser.add_argument('--data_path2', type=str, default="dataset/example_xinjiang.xlsx", help='xinjiang_path')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')

    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes or categories in the dataset')
    parser.add_argument('--features1', type=int, default=11, help='feature')
    parser.add_argument('--features2', type=int, default=10, help='feature')
    parser.add_argument('--weights1', nargs='+', type=float, default=[0.4, 0.5, 0.5, 0.5, 0.4],
                        help='Focal loss weights for dataset 1')
    parser.add_argument('--weights2', nargs='+', type=float, default=[0.2, 0.2, 0.5, 0.1, 0.6],
                        help='Focal loss weights for dataset 2')
    parser.add_argument('--weights3', nargs='+', type=float, default=[0.5, 0.5, 0.4, 0.4, 0.5],
                        help='Focal loss weights for dataset 3')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
def convert_to_tensor(data_list):
    return [torch.tensor(data, dtype=torch.float32) for data in data_list]

class MultiScaleDataset(Dataset):
    def __init__(self, data_by_scale, labels):
        self.data_by_scale = data_by_scale
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return tuple(data[index] for data in self.data_by_scale), self.labels[index]
def prepare_data(data):
    scales = [1, 3, 5]
    new_data_by_scale = [[] for _ in scales]
    for i in range(len(data)):
        new_data_by_scale[0].append(data[i])
        if i == 0:
            new_data_by_scale[1].append(np.vstack([np.zeros(data.shape[1]), data[i], data[min(i + 1, len(data) - 1)]]))
        elif i == len(data) - 1:
            new_data_by_scale[1].append(np.vstack([data[max(i - 1, 0)], data[i], np.zeros(data.shape[1])]))
        else:
            new_data_by_scale[1].append(np.vstack([data[i - 1], data[i], data[i + 1]]))
        left_padding_2 = np.zeros(data.shape[1]) if i - 2 < 0 else data[i - 2]
        left_padding_1 = np.zeros(data.shape[1]) if i - 1 < 0 else data[i - 1]
        right_padding_1 = np.zeros(data.shape[1]) if i + 1 >= len(data) else data[i + 1]
        right_padding_2 = np.zeros(data.shape[1]) if i + 2 >= len(data) else data[i + 2]
        new_data_by_scale[2].append(
            np.vstack([left_padding_2, left_padding_1, data[i], right_padding_1, right_padding_2]))
    new_data_by_scale = convert_to_tensor(new_data_by_scale)
    return new_data_by_scale
def generate_multiscale_data(data, labels):
        new_data_by_scale = prepare_data(data)
        j = 1
        X_train, X_test, y_train, y_test = [], [], [], []
        train_test_data = []
        for i in range(len(new_data_by_scale)):
            reshaped_data = new_data_by_scale[i] \
                .reshape(new_data_by_scale[i].size(0), -1)
            scaler = preprocessing.StandardScaler()
            scaled_data_reshaped = scaler.fit_transform(reshaped_data)
            scaled_data_reshaped = torch.FloatTensor(scaled_data_reshaped)
            if i == 0:
                new_data_by_scale[i] = new_data_by_scale[i].unsqueeze(1)
            else:
                new_data_by_scale[i] = scaled_data_reshaped. \
                    reshape(new_data_by_scale[i].size(0), j, new_data_by_scale[i].size(2))
            j = j + 2
            X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(new_data_by_scale[i], labels,
                                                                                    test_size=0.3,
                                                                                    random_state=42)
            X_train.append(X_train_data)
            X_test.append(X_test_data)
            y_train.append(y_train_data)
            y_test.append(y_test_data)
        train_test_data.append(X_train)
        train_test_data.append(X_test)
        train_test_data.append(y_train)
        train_test_data.append(y_test)
        return train_test_data
def generate_multiscale_blind(data, labels):
    new_data_by_scale = prepare_data(data)
    j=1
    for i in range(len(new_data_by_scale)):
        reshaped_data = new_data_by_scale[i] \
            .reshape(new_data_by_scale[i].size(0), -1)
        scaler = preprocessing.StandardScaler()
        scaled_data_reshaped = scaler.fit_transform(reshaped_data)
        scaled_data_reshaped = torch.FloatTensor(scaled_data_reshaped)
        if i == 0:
            new_data_by_scale[i] = new_data_by_scale[i].unsqueeze(1)
        else:
            new_data_by_scale[i] = scaled_data_reshaped. \
                reshape(new_data_by_scale[i].size(0), j, new_data_by_scale[i].size(2))
        j = j + 2
    return new_data_by_scale,labels
def get_confusion_matrix(trues, preds):
    conf_matrix = confusion_matrix(trues, preds)
    return conf_matrix
def write_file(file_path,predicted):
    with open(file_path, "w") as file:
        for prediction in predicted:
            file.write(f"{prediction.item()}\n")
        file.close()
def save_matrix(conf_matrix_path,conf_matrix):
    with open(conf_matrix_path, 'wb') as f:
        pickle.dump(conf_matrix, f)