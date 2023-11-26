from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import cross_val_score
import pickle
import os
from sklearn.svm import SVC


def my_10_fold(my_dir, filename):

    train_x = []
    train_y = []

    train_data = pd.read_csv(os.path.join(my_dir, filename))
    train_x = train_data.iloc[:, 1:].values
    train_y = train_data.iloc[:, 0].values

    clf = SVC(gamma=0.0001,  C=10, probability=True, kernel='rbf')

    auc = cross_val_score(clf, train_x, train_y, cv=10, scoring='roc_auc')
    accuracy = cross_val_score(
        clf, train_x, train_y, cv=10, scoring='accuracy')
    f = cross_val_score(clf, train_x, train_y, cv=10, scoring='f1_weighted')
    precision = cross_val_score(
        clf, train_x, train_y, cv=10, scoring='precision')
    recall = cross_val_score(clf, train_x, train_y, cv=10, scoring='recall')

    print('10-fold result:')
    print('auc', auc.mean())
    print('accuracy', accuracy.mean())
    print('f-score', f.mean())
    print('precision', precision.mean())
    print('recall', recall.mean())
    print('\n')

    clf.fit(train_x, train_y)
    output = open('./SVMmodel.pkl', 'wb')
    pickle.dump(clf, output)
    output.close()


def test_SVM(my_dir, filename):
    test_x = []
    test_y = []

    test_data = pd.read_csv(os.path.join(my_dir, filename))
    test_x = test_data.iloc[:, 1:].values
    test_y = test_data.iloc[:, 0].values

    input_model = open('./SVMmodel.pkl', 'rb')
    test_model = pickle.load(input_model)

    pred_y = test_model.predict(test_x)
    test_acc = accuracy_score(pred_y, test_y)
    test_roc = metrics.roc_auc_score(test_y, pred_y)

    print('test result:')
    print('accuracy', '%.4f' % test_acc)
    print('auc', '%.4f' % test_roc)
    print(classification_report(test_y, pred_y))


# TODO:
# train folder and test folder here
train_folder = './train_data/'
test_folder = './test_data/'

for file in os.listdir(train_folder):
    print('result at the ' + file[:-4] + ' level')
    result_10fold = my_10_fold(train_folder, file)
    result_test = test_SVM(test_folder, file)
