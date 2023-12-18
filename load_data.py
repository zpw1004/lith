from torch.utils.data import TensorDataset

from util import *
import pandas as pd
def get_daqing(path):
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)
    X_train = data_frame.loc[:,
              ["SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "Face"]]
    X_train_first = X_train.drop(labels='Face', axis=1)
    X= X_train_first.values
    max_min = preprocessing.StandardScaler()
    X = max_min.fit_transform(X_train_first)
    y_train_first = data_frame['Face']
    y = y_train_first.values
    y = torch.LongTensor(y)
    train_test_data = generate_multiscale_data(X, y)
    dataset= MultiScaleDataset(train_test_data[0],train_test_data[2][0])
    data_trainloader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataset_test = MultiScaleDataset(train_test_data[1],train_test_data[3][0])
    data_valloader = DataLoader(dataset_test, batch_size=32, shuffle=False)
    return train_test_data[0],train_test_data[2][0],data_trainloader,train_test_data[1],train_test_data[3][0],data_valloader

def get_xinjiang(path):
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)
    X_train = data_frame.loc[:,
              ["GR", "DT24", "DTC", "DTS", "DTST", "P", "RD", "RS", "TH", "U", "Face"]]
    X_train_first = X_train.drop(labels='Face', axis=1)
    X = X_train_first.values
    max_min = preprocessing.StandardScaler()
    X = max_min.fit_transform(X_train_first)
    y_train_first = data_frame['Face']
    y = y_train_first.values
    y = torch.LongTensor(y)
    train_test_data = generate_multiscale_data(X, y)
    dataset = MultiScaleDataset(train_test_data[0], train_test_data[2][0])
    data_trainloader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataset_test = MultiScaleDataset(train_test_data[1], train_test_data[3][0])
    data_valloader = DataLoader(dataset_test, batch_size=32, shuffle=False)
    return train_test_data[0], train_test_data[2][0], data_trainloader, train_test_data[1], train_test_data[3][
        0], data_valloader

def get_blind(path):
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)
    # X_train_frame = data_frame[(data_frame['Well'] != '宁228') & (data_frame['Well'] != '蛟11')]
    # X_blind_frame = data_frame[(data_frame['Well'] == '蛟11') | (data_frame['Well'] == '宁228')]

    X_train_frame = data_frame[(data_frame['Well'] != 'A')]
    X_blind_frame = data_frame[(data_frame['Well'] == 'C')]
    X_train = X_train_frame.loc[:,
              ["SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "Face"]]
    X_train = X_train.drop(labels='Face', axis=1)
    X_train= X_train.values
    y_train =X_train_frame['Face']
    y_train = y_train.values
    X_blind = X_blind_frame.loc[:,
              ["SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "Face"]]
    X_blind = X_blind.drop(labels='Face', axis=1)
    X_blind = X_blind.values
    y_blind =X_blind_frame['Face']
    y_blind = y_blind.values
    new_data_by_scale_train, new_labels = generate_multiscale_blind(X_train, y_train)
    new_data_by_scale_test, new_labels = generate_multiscale_blind(X_blind, y_blind)
    data_train_y = torch.LongTensor(y_train)
    data_test_y = torch.LongTensor(y_blind)
    dataset= MultiScaleDataset(new_data_by_scale_train,data_train_y)
    data_trainloader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataset_test = MultiScaleDataset(new_data_by_scale_test,data_test_y)
    data_valloader = DataLoader(dataset_test, batch_size=32, shuffle=False)
    return new_data_by_scale_train,data_train_y,data_trainloader,new_data_by_scale_test,data_test_y,data_valloader

def get_xinjiang_raw(path):
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)
    X_train = data_frame.loc[:,
              ["GR", "DT24", "DTC", "DTS", "DTST", "P", "RD", "RS", "TH", "U", "Face"]]
    X_train_first = X_train.drop(labels='Face', axis=1)
    X = X_train_first.values
    max_min = preprocessing.StandardScaler()
    X = max_min.fit_transform(X_train_first)
    y_train_first = data_frame['Face']
    y = y_train_first.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return X_train, X_test, y_train, y_test,train_loader,test_loader

def get_daqing_raw(path):
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)
    X_train = data_frame.loc[:,
              ["SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "Face"]]
    X_train_first = X_train.drop(labels='Face', axis=1)
    X = X_train_first.values
    max_min = preprocessing.StandardScaler()
    X = max_min.fit_transform(X_train_first)
    y_train_first = data_frame['Face']
    y = y_train_first.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return X_train, X_test, y_train, y_test, train_loader, test_loader

def get_blind_raw(path):
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)
    # X_train_frame = data_frame[(data_frame['Well'] != '宁228') & (data_frame['Well'] != '蛟11')]
    # X_blind_frame = data_frame[(data_frame['Well'] == '蛟11') | (data_frame['Well'] == '宁228')]
    X_train_frame = data_frame[(data_frame['Well'] != 'A')]
    X_blind_frame = data_frame[(data_frame['Well'] == 'C')]
    X_train = X_train_frame.loc[:,
              ["SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "Face"]]
    X_train = X_train.drop(labels='Face', axis=1)
    X_train = X_train.values
    y_train = X_train_frame['Face']
    y_train = y_train.values
    X_blind = X_blind_frame.loc[:,
              ["SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "Face"]]
    X_blind = X_blind.drop(labels='Face', axis=1)
    X_blind = X_blind.values
    y_blind = X_blind_frame['Face']
    y_blind = y_blind.values
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_blind = torch.tensor(X_blind)
    y_blind = torch.tensor(y_blind)
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataset = TensorDataset(X_blind, y_blind)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return X_train, X_blind, y_train, y_blind, train_loader, test_loader