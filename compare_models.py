import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from utils import save_to_csv_concat
from sklearn.metrics import mean_squared_error



clfs_names = [
    'KNN',
    # 'Decision tree',
    # 'Random Forest',
    # 'SVM',
    # 'MLP',
    # 'Naive Bayes',
    # 'Perceptron'
]


classifiers = [
    KNeighborsClassifier(algorithm='ball_tree', n_neighbors=3, p=1, weights='distance'),
    # DecisionTreeClassifier(criterion='gini', min_samples_leaf=1, min_samples_split=5, splitter='best'),
    # RandomForestClassifier(bootstrap=False, criterion='entropy', min_samples_leaf=1, min_samples_split=3,
    #                        n_estimators=40),
    # SVC(C=100, gamma='auto', kernel='rbf'),
    # MLPClassifier(activation='tanh', max_iter=400, solver='adam'),
    # GaussianNB(),
    # Perceptron(alpha=1e-05, penalty=None)
]


train_data = pd.read_csv('train_final_data_16_17.csv')
X_train = train_data.drop(['tag','patient num'], axis=1)
y_train = train_data.tag

val_data = pd.read_csv('val_final_data_16_17.csv')
X_val = val_data.drop(['tag','patient num'], axis=1)
y_val = val_data.tag


# the measurements
for i, clf in enumerate(classifiers):

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_val)
    print(len(predictions))
    print(len(y_val))
    print(type(y_val))
    print(type(predictions))
    y_val_copy = y_val.copy()
    y_val_copy[y_val_copy <= 365] = 1
    y_val_copy[y_val_copy > 365] = -1
    print(y_val_copy)
    print(y_val)
    # for k in range(len(predictions)):
    #     print(k)
    #     print(predictions[k], y_val[k])

    # f1 = np.mean(f1_score(y_val, predictions, average='macro'))
    # print("clf ", clfs_names[i], " f1 score is: ", f1)
    #
    # accuracy = np.mean(accuracy_score(y_val, predictions))
    # print("clf ", clfs_names[i], " accuracy score is: ", accuracy)
    #
    # print("clf ", clfs_names[i], "total score is ", (f1+accuracy)/2)
    #
    # mse = mean_squared_error(y_val, predictions)
    # print("clf ", clfs_names[i], " mse score is: ", mse)
