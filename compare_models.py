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
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
from utils import save_to_csv_concat
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter
from lifelines import LogLogisticAFTFitter
from lifelines import LogNormalAFTFitter
from lifelines import AalenAdditiveFitter
from sklearn.metrics import average_precision_score, precision_score
from sklearn.metrics import recall_score
import utils


clfs_names = [
    'logistic regression',
    'linear regression',
    'KNN'
    'Decision tree',
    'Random Forest',
    'SVM',
    'MLP',
    'Naive Bayes',
    'Perceptron'
]


classifiers = [
    LogisticRegression(),
    LinearRegression(),
    KNeighborsClassifier(algorithm='ball_tree', n_neighbors=3, p=1, weights='distance'),
    DecisionTreeClassifier(criterion='gini', min_samples_leaf=1, min_samples_split=5, splitter='best'),
    RandomForestClassifier(bootstrap=False, criterion='entropy', min_samples_leaf=1, min_samples_split=3,
                           n_estimators=40),
    SVC(C=100, gamma='auto', kernel='rbf'),
    MLPClassifier(activation='tanh', max_iter=400, solver='adam'),
    GaussianNB(),
    Perceptron(alpha=1e-05, penalty=None)
]


def prepare_cph_train_data(data):
    data['predictions'] = [1 if (x <= 365) and (x > 0) else 0 for x in data['tag']]
    data['duration'] = [data['tag'][i] if (data['tag'][i] <= 365) else 365 for i
                                     in range(len(data['tag']))]
    c = data.columns[data.eq(data.iloc[0]).all()].tolist()
    data = data.drop(['tag', 'patient num'], axis=1)
    data = data.drop(c, axis=1)
    return data, c


def prepare_aft_train_data(data):
    aft_train = data.copy()
    aft_train['duration'] = [aft_train['duration'][i] if (aft_train['duration'][i] > 0) else 0.00001 for i in
                             range(len(aft_train['duration']))]
    return aft_train


def prepare_cph_val_data(data, c):
    data = data.drop(c, axis=1)
    data = data.drop(['tag', 'patient num'], axis=1)
    return data


def predictions_to_binary(predictions):
    predictions_copy = predictions.copy()
    predictions_copy[predictions_copy <= 365] = 1
    predictions_copy[predictions_copy > 365] = 0
    return predictions_copy


def evaluation(predictions_binary, predictions, model_name, y_val_binary, y_val):
    f1 = np.mean(f1_score(y_val_binary, predictions_binary, average='macro'))
    print(model_name, " f1 score is: ", f1)
    accuracy = np.mean(accuracy_score(y_val_binary, predictions_binary))
    print(model_name, " accuracy score is: ", accuracy)
    print(model_name, " total score is ", (f1 + accuracy) / 2)
    mae = mean_absolute_error(y_val_binary, predictions_binary)
    print(model_name, " MAE score is ", mae)
    mae = mean_absolute_error(y_val, predictions)
    print(model_name, " MAE score not binary is ", mae)
    precision = precision_score(y_val_binary, predictions_binary)
    print(model_name, " percision score is ", precision)
    recall = recall_score(y_val_binary, predictions_binary)
    print(model_name, " recall score is ", recall)


# train preparation
train_data = pd.read_csv('train_final_data_16_17.csv')
cph_train, c = prepare_cph_train_data(train_data)
aft_train = prepare_aft_train_data(cph_train)

feature_data = pd.read_csv('feature_selection_16_17.csv')
cph_train_feature, c_feature = prepare_cph_train_data(feature_data)
cph_train_feature['test date'] = train_data['test date']
aft_train_feature = prepare_aft_train_data(cph_train_feature)

var_data = pd.read_csv('train_final_with_var_16_17.csv')
cph_train_var, c_var = prepare_cph_train_data(var_data)
cph_train_var = cph_train_var.drop(['AFP (Alpha fetoprotein) - bloo'], axis=1)
aft_train_var = prepare_aft_train_data(cph_train_var)
#aft_train_var = aft_train_var.drop(['AFP (Alpha fetoprotein) - bloo binary', 'ANA 1:640', 'ANA 1:640 binary', 'APTT FS','APTT FS binary', 'AT'], axis=1)

cluster_data = pd.read_csv('train_final_cluster_16_17.csv')
cph_train_cluster, c_cluster = prepare_cph_train_data(cluster_data)
aft_train_cluster = prepare_aft_train_data(cph_train_cluster)

cluster_data_3 = pd.read_csv('train_final_cluster_k3_16_17.csv')
cph_train_cluster_3, c_cluster_3 = prepare_cph_train_data(cluster_data_3)



# cph_train_feature = feature_data.copy()
# cph_train_var = var_data.copy()
# cph_train = train_data.copy()
# cph_train_cluster = cluster_data.copy()
# cph_train = cph_train.drop('patient num', 1)
# cph_train_feature['predictions'] = [1 if (x <= 365) and (x > 0) else 0 for x in cph_train_feature['tag']]
# cph_train_feature['duration'] = [cph_train_feature['tag'][i] if (cph_train_feature['tag'][i] <= 365) else 365 for i in range(len(cph_train_feature['tag']))]
# cph_train_var['predictions'] = [1 if (x <= 365) and (x > 0) else 0 for x in cph_train_var['tag']]
# cph_train_var['duration'] = [cph_train_var['tag'][i] if (cph_train_var['tag'][i] <= 365) else 365 for i in range(len(cph_train_var['tag']))]
# cph_train['predictions'] = [1 if (x <= 365) and (x > 0) else 0 for x in cph_train['tag']]
# cph_train['duration'] = [cph_train['tag'][i] if (cph_train['tag'][i] <= 365) else 365 for i in range(len(cph_train['tag']))]
# cph_train_cluster['predictions'] = [1 if (x <= 365) and (x > 0) else 0 for x in cph_train_cluster['tag']]
# cph_train_cluster['duration'] = [cph_train_cluster['tag'][i] if (cph_train_cluster['tag'][i] <= 365) else 365 for i in range(len(cph_train_cluster['tag']))]
# c = cph_train.columns[cph_train.eq(cph_train.iloc[0]).all()].tolist()
# c_feature = cph_train_feature.columns[cph_train_feature.eq(cph_train_feature.iloc[0]).all()].tolist()
# c_var = cph_train_var.columns[cph_train_var.eq(cph_train_var.iloc[0]).all()].tolist()
# c_cluster = cph_train_cluster.columns[cph_train_cluster.eq(cph_train_cluster.iloc[0]).all()].tolist()
# cph_train = cph_train.drop(c, axis=1)
# cph_train_feature = cph_train_feature.drop(c_feature, axis=1)
# cph_train_var = cph_train_var.drop(c_var, axis=1)
# cph_train_feature = cph_train_feature.drop(['tag', 'patient_num'], axis=1)
# cph_train_cluster = cph_train_cluster.drop(c_cluster, axis=1)
# cph_train_cluster = cph_train_cluster.drop(['tag', 'patient num'], axis=1)
# cph_train = cph_train.drop(['tag', 'patient num'], axis=1)
# X_train = train_data.drop(['tag'], axis=1)
# y_train = train_data.tag

# val preparation
val_data = pd.read_csv('val_final_data_16_17.csv')
cph_val = prepare_cph_val_data(val_data, c)

data_var = pd.read_csv('val_final_with_var_16_17.csv')
cph_val_var = prepare_cph_val_data(data_var, c_var)
cph_val_var = cph_val_var.drop(['AFP (Alpha fetoprotein) - bloo'], axis=1)
#aft_val_var = cph_val_var.drop(['AFP (Alpha fetoprotein) - bloo binary', 'ANA 1:640', 'ANA 1:640 binary', 'APTT FS', 'APTT FS binary', 'AT'], axis=1)

val_cluster = pd.read_csv('val_final_cluster_16_17.csv')
cph_val_cluster = prepare_cph_val_data(val_cluster, c_cluster)

val_cluster_3 = pd.read_csv('val_final_cluster_k3_16_17.csv')
cph_val_cluster_3 = prepare_cph_val_data(val_cluster_3, c_cluster_3)

selected_features = list(feature_data)
val_data_feature = pd.DataFrame(val_data, columns=selected_features)
cph_val_feature = prepare_cph_val_data(val_data_feature, c_feature)
cph_val_feature['test date'] = val_data['test date']

# X_val = val_data.drop(['tag'], axis=1)
# X_val_copy = val_data.drop(['tag'], axis=1)
# cph_val_feature = val_data_feature.drop(['tag', 'patient_num'], axis=1)
# cph_val_feature = cph_val_feature.drop(c_feature, axis=1)
# cph_val_cluster = val_cluster.drop(['tag', 'patient num'], axis=1)
# cph_val_cluster = val_cluster.drop(c_cluster, axis=1)
# cph_val_var = cph_val_var.drop(c_var, axis=1)
# cph_val = val_data.drop(['tag', 'patient num'], axis=1)
# cph_val = cph_val.drop(c, axis=1)

# y_val preparation
y_val = val_data.tag
y_val_copy = y_val.copy()
y_val_copy[y_val_copy <= 365] = 1
y_val_copy[y_val_copy > 365] = 0


# def run_all_classifiers():
#     # the measurements
#     for i, clf in enumerate(classifiers):
#         clf.fit(X_train, y_train)
#         predictions = clf.predict(X_val)
#         # print(len(predictions))
#         # print(len(y_val))
#         # print(type(y_val))
#         # print(type(predictions))
#         y_val_copy = y_val.copy()
#         y_val_copy[y_val_copy <= 365] = 1
#         y_val_copy[y_val_copy > 365] = 0
#         # print(y_val_copy)
#         # print(y_val)
#         predictions_copy = predictions.copy()
#         predictions_copy[predictions_copy <= 365] = 1
#         predictions_copy[predictions_copy > 365] = 0
#         # print(predictions)
#         # print('###################')
#         # print(predictions_copy)
#
#         # for k in range(len(predictions)):
#         #     print(k)
#         #     print(predictions_copy[k], y_val_copy[k])
#
#         f1 = np.mean(f1_score(y_val_copy, predictions_copy, average='macro'))
#         print("clf ", clfs_names[i], " f1 score is: ", f1)
#
#         accuracy = np.mean(accuracy_score(y_val_copy, predictions_copy))
#         print("clf ", clfs_names[i], " accuracy score is: ", accuracy)
#
#         print("clf ", clfs_names[i], "total score is ", (f1 + accuracy) / 2)
#
#         mae = mean_absolute_error(y_val, predictions_copy)
#         print("clf ", clfs_names[i], "MAE score is ", mae)
#         mse = mean_squared_error(y_val_copy, predictions_copy)
#         print("clf ", clfs_names[i], " mse score is: ", mse)


def run_cox_fitter(train, val):
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train, duration_col='duration', event_col='predictions', show_progress=True, step_size=0.5)
    predictions = cph.predict_expectation(val)
    # utils.save_to_csv(predictions, "cox_predictions.csv")
    predictions_binary = predictions_to_binary(predictions)
    # utils.save_to_csv(predictions_binary, "cox_predictions_binary.csv")
    evaluation(predictions_binary, predictions, "Cox fitter", y_val_copy, y_val)
    # cph_train["predicted"] = predictions
    # print("######only dead!#####")
    # predictions_dead = cph_train[cph_train["predicted"] <= 365]
    # predictions_dead_binary = predictions_to_binary(predictions_dead)
    # predictions_dead_indexes = cph_train.index[cph_train["predicted"] <= 365].tolist()
    # y_val_dead = val_data[val_data.index.isin(predictions_dead_indexes)]
    # y_val_dead = y_val_dead.tag
    # y_val_dead_copy = y_val_dead.copy()
    # y_val_dead_copy[y_val_dead_copy <= 365] = 1
    # y_val_dead_copy[y_val_dead_copy > 365] = 0
    # evaluation(predictions_dead_binary, predictions_dead, "Cox fitter", y_val_dead_copy, y_val_dead)


def run_aft_fitter(train, val):
    aft = WeibullAFTFitter()
    aft._scipy_fit_options = {"maxiter": 500}
    aft.fit(train, duration_col='duration', event_col='predictions', ancillary_df=None)
    predictions_weibull_aft = aft.predict_expectation(val)
    predictions_binary = predictions_to_binary(predictions_weibull_aft)
    evaluation(predictions_binary, predictions_weibull_aft, "Weibull AFT fitter", y_val_copy, y_val)


def run_log_logistic_aft_fitter(train, val):
    aft = LogLogisticAFTFitter()
    aft._scipy_fit_options = {"maxiter": 500}
    aft.fit(train, duration_col='duration', event_col='predictions', show_progress=True)
    predictions_log_logistic_aft = aft.predict_expectation(val)
    utils.save_to_csv(predictions_log_logistic_aft, "predictions_log_logistic.csv")
    predictions_binary = predictions_to_binary(predictions_log_logistic_aft)
    utils.save_to_csv(predictions_binary, "predictions_log_logistic_binary.csv")
    evaluation(predictions_binary, predictions_log_logistic_aft, "Log logistic AFT fitter", y_val_copy, y_val)


def run_log_normal_aft_fitter(train, val):
    aft = LogNormalAFTFitter()
    aft.fit(train, duration_col='duration', event_col='predictions')
    predictions_log_normal_aft = aft.predict_expectation(val)
    predictions_binary = predictions_to_binary(predictions_log_normal_aft)
    evaluation(predictions_binary, predictions_log_normal_aft, "Log normal AFT fitter", y_val_copy, y_val)


def run_aalen_additive_fitter(train, val):
    aaf = AalenAdditiveFitter(coef_penalizer=1.0, fit_intercept=False)
    aaf.fit(train, 'duration', event_col='predictions')
    predictions_aaf = aaf.predict_expectation(val)
    predictions_binary = predictions_to_binary(predictions_aaf)
    evaluation(predictions_binary, predictions_aaf, "Aalen Additive fitter", y_val_copy, y_val)


# print("#######regular data#######")
# run_cox_fitter(cph_train, cph_val)
# run_aalen_additive_fitter(cph_train, cph_val)
# run_aft_fitter(aft_train, cph_val)
# run_log_normal_aft_fitter(aft_train, cph_val)
# run_log_logistic_aft_fitter(aft_train, cph_val)

# print("#######feature selected data#######")
# run_cox_fitter(cph_train_feature, cph_val_feature)
# run_aalen_additive_fitter(cph_train_feature, cph_val_feature)
# run_aft_fitter(aft_train_feature, cph_val_feature)
# run_log_normal_aft_fitter(aft_train_feature, cph_val_feature)
# run_log_logistic_aft_fitter(aft_train_feature, cph_val_feature)

# print("#######clustered data#########")
run_cox_fitter(cph_train_cluster_3, cph_val_cluster_3)
# run_cox_fitter(cph_train_cluster, cph_val_cluster)
# run_aalen_additive_fitter(cph_train_cluster, cph_val_cluster)
# run_aft_fitter(aft_train_cluster, cph_val_cluster)
# run_log_normal_aft_fitter(aft_train_cluster, cph_val_cluster)
# run_log_logistic_aft_fitter(aft_train_cluster, cph_val_cluster)

# print("#####data with var######")
# run_cox_fitter(cph_train_var, cph_val_var)
# run_aalen_additive_fitter(cph_train_var, cph_val_var)
# run_aft_fitter(aft_train_var, aft_val_var)
# run_log_normal_aft_fitter(aft_train_var, aft_val_var)
# run_log_logistic_aft_fitter(aft_train_var, aft_val_var)


#get prediction of only dead people
# cph = CoxPHFitter(penalizer=0.1)
# # cph.fit(cph_train, duration_col='duration', event_col='predictions', show_progress=True, step_size=0.5)
# # predictions = cph.predict_expectation(cph_val)
# predictions = pd.read_csv("predictions.csv")
# # utils.save_to_csv(predictions, "predictions.csv")
# X_val_copy = val_data.drop(['tag'], axis=1)
# X_val_copy["predict"] = predictions
# predictions_dead = X_val_copy[X_val_copy["predict"] <= 365]
# print("predictions_dead len ", len(predictions_dead))
# sum = 0
# for predict in predictions:
#     if predict <= 365:
#         sum = sum + 1
# print("sum is ", sum)
# predictions_dead_indexes = X_val_copy.index[X_val_copy["predict"] <= 365].tolist()
# y_val_dead = val_data[val_data.index.isin(predictions_dead_indexes)]
# y_val_dead = y_val_dead.tag
#
# #get binary prediction of dead people
# predictions_dead_copy = predictions_dead.copy()
# predictions_dead_copy[predictions_dead_copy <= 365] = 1
# predictions_dead_copy[predictions_dead_copy > 365] = 0
# predictions_copy = predictions_to_binary(predictions)
# y_val_dead_copy = y_val_dead.copy()
# y_val_dead_copy[y_val_dead_copy <= 365] = 1
# y_val_dead_copy[y_val_dead_copy > 365] = 0
# predictions_dead = predictions_dead["predict"]
# predictions_dead_copy = predictions_dead_copy["predict"]
#
#
#
# # evaluation ##
#
#
# f1 = np.mean(f1_score(y_val_dead_copy, predictions_dead_copy, average='macro'))
# print("cph f1 of only alive score is: ", f1)
#
#
# accuracy = np.mean(accuracy_score(y_val_dead_copy, predictions_dead_copy))
# print("cph accuracy only dead score is: ", accuracy)
#
# mae = mean_absolute_error(y_val_dead, predictions_dead)
# print("cph MAE score not binary is ", mae)
#
# average_precision = precision_score(y_val_dead_copy, predictions_dead_copy)
# print("cph only dead percision score is ", average_precision)
#
# average_precision = precision_score(y_val_copy, predictions_copy)
# print("cph percision score is ", average_precision)
#
# recall = recall_score(y_val_dead_copy, predictions_dead_copy)
# print("cph only dead recall score is ", recall)





