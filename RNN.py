import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

# Set True if want to use bidirectional RNN
BIDIRECTIONAL = False


class RNNDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index].values


class TFDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        assert len(self.data) == len(self.label)
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class SimpleRNN(nn.Module):
    def __init__(self, input_len, hidden_size, output_size, bidirectional=False):
        super(SimpleRNN, self).__init__()
        self.hidden_szie = hidden_size
        self.bidirectional = bidirectional

        self.rnn = nn.RNN(input_len, hidden_size, batch_first=True,
                          bidirectional=self.bidirectional)
        if bidirectional:
            hidden_size = 2 * hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.sig1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sig2 = nn.Sigmoid()

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        if self.bidirectional:
            dim0 = 2
        else:
            dim0 = 1
        hidden = torch.zeros(dim0, x.size(0), self.hidden_szie)
        out, hidden = self.rnn(x, hidden)
        out = out[:, -1, :]
        out = self.sig1(self.fc1(out))
        out = self.sig2(self.fc2(out))
        return out


def train_modelTF(model, train_dataset, train_label, optimizer,
                  criterion, epoch_cnt, batch_szie):
    data_loader = DataLoader(
        TFDataset(train_dataset, train_label), batch_size=batch_szie, shuffle=True)
    for _ in range(epoch_cnt):
        epoch_loss = []
        for (data, label) in data_loader:
            input = data.to(dtype=torch.float32)
            output = model(input)
            loss = criterion(output, label.reshape(-1, 1).to(torch.float32))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())


def test_model(model, X_test, y, result_dict: dict, name, append_as_TF=False):
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_pred = model(X_test).detach().numpy()
    y_pred_bin = y_pred > 0.5
    auc = roc_auc_score(y, y_pred)
    f1 = f1_score(y, y_pred_bin)
    precesion = precision_score(y, y_pred_bin)
    recall = recall_score(y, y_pred_bin)
    accuracy = accuracy_score(y, y_pred_bin)

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
input_size = 256
output_size = 1
hidden_size = 20
lr = 0.001
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
    # for tenfold validation only
    X_train = train_data.iloc[:, 1:].values
    train_label = train_data.iloc[:, 0].values

    test_data = pd.read_csv(os.path.join(test_folder, train_file))
    X_test = test_data.iloc[:, 1:].values
    test_labels = test_data.iloc[:, 0].values
    # tenfold cross validation
    print("validation")
    for train_indices, test_indices in tqdm(kfold.split(X_train, train_label),
                                            desc="tenfold", total=10, colour='blue'):

        model = SimpleRNN(input_size, hidden_size, output_size,
                          bidirectional=BIDIRECTIONAL)

        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr)

        tf_train, label_train = X_train[train_indices], train_label[train_indices]
        tf_val, label_val = X_train[test_indices], train_label[test_indices]

        train_modelTF(model, tf_train, label_train, optimizer,
                      criterion, num_epochs, batch_szie)
        test_model(model, tf_val, label_val,
                   tenfold_dict, name, append_as_TF=True)

    model = SimpleRNN(input_size, hidden_size, output_size,
                      bidirectional=BIDIRECTIONAL)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr)
    data_loader = DataLoader(RNNDataset(train_data),
                             batch_size=batch_szie, shuffle=True)

    print("train for final test")
    for epoch in tqdm(range(num_epochs), desc="epoch", colour='green'):
        epoch_loss = []
        for batch_data in data_loader:
            labels = batch_data[:, 0]

            inputs = batch_data[:, 1:].to(dtype=torch.float32)
            outputs = model(inputs)
            loss = criterion(outputs, labels.reshape(-1, 1).to(torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
    test_model(model, X_test, test_labels, test_data_dict, name)

    print("tenfold")
    print_dict_content(tenfold_dict, need_avg=True)
    print("test")
    print_dict_content(test_data_dict)
