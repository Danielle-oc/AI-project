import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV, KFold


train_data = pd.read_csv('train_final_data_16_17.csv')
X_train = train_data.drop('tag', axis=1)
y_train = train_data.tag

cv = KFold(n_splits=10, shuffle=True)


def knn_tuning(data, cv):
    clf = KNeighborsClassifier()
    k_list = [1, 3, 5, 7, 9, 11, 13]
    params = {
        'weights': ['distance', 'uniform'],
        'n_neighbors': k_list,
        'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
        'p': [1, 2, 3, 4]
    }
    grid = GridSearchCV(clf, param_grid=params, cv=cv)
    grid.fit(data.drop('tag', 1), data['tag'])
    print("best params for knn are %s, score is: %0.2f" % (grid.best_params_, grid.best_score_))
    return grid.best_params_


def decision_tree_tuning(data, cv):
    clf = DecisionTreeClassifier()
    params = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'min_samples_split': [3, 5, 7, 9, 11, 13, 15, 17, 19],
        'min_samples_leaf': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    }
    grid = GridSearchCV(clf, param_grid=params, cv=cv)
    grid.fit(data.drop('tag', 1), data['tag'])
    print("best params for decision tree are %s, score is: %0.2f" % (grid.best_params_, grid.best_score_))
    return grid.best_params_


def random_forest_tuning(data, cv):
    clf = RandomForestClassifier()
    params = {
        'n_estimators': np.arange(10, 50, 10),
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [3, 5, 7, 9, 11, 13, 15, 17, 19],
        'min_samples_leaf': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        'bootstrap': [True, False]
    }
    grid = GridSearchCV(clf, param_grid=params, cv=cv)
    grid.fit(data.drop('tag', 1), data['tag'])
    print("best params for random forest are %s, score is: %0.2f" % (grid.best_params_, grid.best_score_))
    return grid.best_params_


def svc_tuning(data, cv):
    clf = SVC()

    params = [
        {'C': [0.1, 1, 10, 100], 'kernel': ['linear']},
        {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']},
        {'C': [0.1, 1, 10, 100], 'degree': [2, 3, 4, 5], 'kernel': ['poly']},
    ]

    grid = GridSearchCV(clf, param_grid=params, cv=cv)
    grid.fit(data.drop('tag', 1), data['tag'])
    print("best params for svc are %s, score is: %0.2f" % (grid.best_params_, grid.best_score_))
    return grid.best_params_


def mlp_tuning(data, cv):
    clf = MLPClassifier()
    params = {
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'max_iter': [200, 400, 600],
    }
    grid = GridSearchCV(clf, param_grid=params, cv=cv)
    grid.fit(data.drop('tag', 1), data['tag'])
    print("best params for mlp are %s, score is: %0.2f" % (grid.best_params_, grid.best_score_))
    return grid.best_params_


def perceptron_tuning(data, cv):
    clf = Perceptron()
    params = {
        'penalty': [None, 'l2', 'l1', 'elasticnet'],
        'alpha': [0.00001, 0.0001, 0.001, 0.01]
    }
    grid = GridSearchCV(clf, param_grid=params, cv=cv)
    grid.fit(data.drop('tag', 1), data['tag'])
    print("best params for perceptron are %s, score is: %0.2f" % (grid.best_params_, grid.best_score_))
    return grid.best_params_


knn_tuning(train_data, cv)
decision_tree_tuning(train_data, cv)
random_forest_tuning(train_data, cv)
mlp_tuning(train_data, cv)
perceptron_tuning(train_data, cv)
svc_tuning(train_data, cv)
