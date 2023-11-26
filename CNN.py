import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import os
from tabulate import tabulate
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score


class CNNDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class CNN1d(nn.Module):
    def __init__(self):
        super(CNN1d, self).__init__()
        self.conv1d = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # 256 * 2
        self.maxpool = nn.MaxPool1d(kernel_size=4)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train_model(model, train_dataset, train_label, optimizer,
                criterion, epoch_cnt, batch_size):
    data_loader = DataLoader(CNNDataSet(
        train_dataset, train_label), batch_size=batch_size, shuffle=True)
    for _ in range(epoch_cnt):
        epoch_loss = []
        for (data, label) in data_loader:
            input = data.to(dtype=torch.float32)
            output = model(input)
            labels_onehot = F.one_hot(
                label, num_classes=2).to(dtype=torch.float32)
            loss = criterion(output, labels_onehot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())


def test_model(model, X_test, y, result_dict: dict, name, append_as_TF=False):
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_pred = model(X_test)
    y_pred = torch.argmax(y_pred, dim=1)

    auc = roc_auc_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    precesion = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)

    if append_as_TF:
        if name in result_dict.keys():
            result_dict[name]["AUC"].append(auc)
            result_dict[name]["F1"].append(f1)
            result_dict[name]["Precision"].append(precesion)
            result_dict[name]["Recall"].append(recall)
            result_dict[name]["Accuracy"].append(accuracy)
        else:
            result_dict[name] = {
                "AUC": [auc],
                "F1": [f1],
                "Precision": [precesion],
                "Recall": [recall],
                "Accuracy": [accuracy],
            }
    else:
        result_dict[name] = {
            "AUC": auc,
            "F1": f1,
            "Precision": precesion,
            "Recall": recall,
            "Accuracy": accuracy,
        }


def print_dict_content(dict: dict, need_avg=False):
    result = []
    for name in dict.keys():
        val_map = dict[name]

        for key, value in val_map.items():
            if not need_avg:
                result.append([key, value])
            else:
                result.append([key, sum(value)/len(value)])
    headers = ["meteric", "value"]
    print(tabulate(result, headers=headers, tablefmt="fancy_grid"))



# TODO:
# test folder and train folder here
test_folder = './test_data'
train_folder = './train_data'
lr = 0.002
num_epochs = 25
batch_szie = 16



kfold = StratifiedKFold(n_splits=10, shuffle=True)

for train_file in os.listdir(train_folder):
    test_data_dict = {}
    tenfold_dict = {}

    if train_file.startswith("exp"):
        continue

    name = train_file.split("_")[0]

    print(f'this is file {train_file}')
    train_data = pd.read_csv(os.path.join(train_folder, train_file))
    X_train = train_data.iloc[:, 1:].values
    train_label = train_data.iloc[:, 0].values

    test_data = pd.read_csv(os.path.join(test_folder, train_file))
    X_test = test_data.iloc[:, 1:].values
    test_labels = test_data.iloc[:, 0].values

    print("train for tenfold")
    for train_indices, test_indices in tqdm(kfold.split(X_train, train_label), 
                                            desc='cross validiation', total=10, colour='green'):
        model = CNN1d()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr)

        tf_train, label_train = X_train[train_indices], train_label[train_indices]
        tf_val, label_val = X_train[test_indices], train_label[test_indices]

        train_model(model, tf_train, label_train, optimizer,
                    criterion, num_epochs, batch_szie)
        test_model(model, tf_val, label_val,
                   tenfold_dict, name, append_as_TF=True)

    model = CNN1d()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr)

    print("train for final test")
    train_model(model, X_train, train_label, optimizer,
                criterion, num_epochs, batch_szie)
    test_model(model, X_test, test_labels, test_data_dict, name)


    print("tenfold")
    print_dict_content(tenfold_dict, need_avg=True)
    print("test")
    print_dict_content(test_data_dict)
