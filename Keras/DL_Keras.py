import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Bidirectional
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold

import os

# testing RNN, BiRNN（Based on keras） with TenFold and testset

# Folders here
# test files and Train Filess
TEST_FOLDER = '/Users/dinghao/Desktop/PBLKS_other/keras_data/test'
TRAIN_FOLER = '/Users/dinghao/Desktop/PBLKS_other/keras_data/train'


def get_data(folder: str, file_name: str) -> (np.ndarray, np.ndarray):
    file_path = os.path.join(folder, file_name)
    data = pd.read_csv(file_path)
    feas = data.iloc[:, 1:].values
    labels = data.iloc[:, 0].values
    return (feas, labels)


def eval_model(model: Sequential,
               X_test: np.ndarray,
               y: np.ndarray) -> dict[str, float]:
    '''
    evaluates the model by all metrics 
    including Accuracy Recall Specificity Precision F-score ROC-AUC
    '''
    y_pred = model.predict(X_test)
    y_pred_bin = y_pred > 0.5

    roc_auc = roc_auc_score(y, y_pred)
    f1 = f1_score(y, y_pred_bin)
    precesion = precision_score(y, y_pred_bin)
    recall = recall_score(y, y_pred_bin)
    accuracy = accuracy_score(y, y_pred_bin)

    cm = confusion_matrix(y, y_pred_bin)
    tn, fp, _, _ = cm.ravel()
    specificity = tn / (tn + fp)

    return {"AUC": roc_auc,
            "F1": f1,
            "precesion": precesion,
            "recall": recall,
            "accuracy": accuracy,
            "Specificity": specificity}


def getRNN(bidirectional: bool) -> Sequential:
    hidden_size = 20
    model = Sequential()
    if bidirectional:
        hidden_size = 2 * hidden_size
        model.add(Bidirectional(
            SimpleRNN(units=hidden_size), input_shape=(1, 256)))
    else:
        model.add(SimpleRNN(units=hidden_size, input_shape=(1, 256)))
    model.add(Dense(units=hidden_size, activation='sigmoid'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['binary_accuracy'])
    return model


def add_score(original_dict: dict[str, list], new_dict: dict[str, int]):
    for key in new_dict:
        if key in original_dict.keys():
            original_dict[key].append(new_dict[key])
        else:
            original_dict[key] = [new_dict[key]]


def make_mean(dict: dict[str, list[float]]) -> dict[str, float]:
    for key in dict:
        dict[key] = np.mean(dict[key])
    return dict


for name in os.listdir(TEST_FOLDER):
    if name.count("test") != 0:
        continue

    print("checking file " + name)
    train_file = os.path.join(TRAIN_FOLER, name)
    test_file = os.path.join(TEST_FOLDER, name)

    X_train, y_train = get_data(TRAIN_FOLER, name)

# TenFold tests
    tenfold = StratifiedKFold(n_splits=10, shuffle=True)
    print("doing 10-fold cross validation")
    CNN_score = {}
    RNN_score = {}
    BiRNN_score = {}
    for train_indices, test_indices in tenfold.split(X_train, y_train):
        tf_train, label_train = X_train[train_indices], y_train[train_indices]
        tf_val, label_val = X_train[test_indices], y_train[test_indices]

        RNN_model = getRNN(False)
        RNN_model.fit(tf_train, label_train, epochs=30, batch_size=16)
        RNN_result_dict = eval_model(RNN_model, tf_val, label_val)
        add_score(RNN_score, RNN_result_dict)
        print("tf RNN")
        print(make_mean(RNN_score))

        BiRNN_model = getRNN(True)
        BiRNN_model.fit(tf_train, label_train, epochs=30, batch_size=16)
        BiRNN_result_dict = eval_model(BiRNN_model, tf_val, label_val)
        add_score(BiRNN_score, BiRNN_result_dict)
        print("tf BiRNN")
        print(make_mean(BiRNN_score))

    print("doing test!")
    X_test, y_test = get_data(TEST_FOLDER, name)

    RNN_model = getRNN(False)
    RNN_model.fit(X_train, y_train)
    print("RNN:")
    print(eval_model(RNN_model, X_test, y_test))

    BiRNN_model = getRNN(True)
    BiRNN_model.fit(X_train, y_train)
    print("BiRNN:")
    print(eval_model(BiRNN_model, X_test, y_test))
