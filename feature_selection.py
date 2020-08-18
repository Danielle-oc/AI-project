import pandas as pd
import utils
from lifelines import CoxPHFitter
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt



train_data = pd.read_csv('train_final_data_16_17.csv')
cph_train = train_data.copy()
cph_train = cph_train.drop('patient num', 1)
cph_train['predictions'] = [1 if (x <= 365) and (x > 0) else 0 for x in cph_train['tag']]
cph_train['duration'] = [cph_train['tag'][i] if (cph_train['tag'][i] <= 365) else 365 for i in range(len(cph_train['tag']))]
c = cph_train.columns[cph_train.eq(cph_train.iloc[0]).all()].tolist()
cph_train = cph_train.drop(c, axis=1)
cph_train = cph_train.drop('tag', 1)
X_train = train_data.copy()
X_train = X_train.drop(['tag','patient num'], axis=1)
y_train = train_data.tag

#cph preperation
val_data = pd.read_csv('val_final_data_16_17.csv')
X_val = val_data.drop(['tag', 'patient num'], axis=1)
X_val_copy = val_data.drop(['tag'], axis=1)
cph_val = val_data.drop(['tag'], 1)
cph_val = cph_val.drop(c, axis=1)
y_val = val_data.tag

def run_cox_fitter(data, val):
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(data, duration_col='duration', event_col='predictions', show_progress=True, step_size=0.5)
    predictions = cph.predict_expectation(cph_val)
    return predictions

def run_feature_selction():
    prev_accuracy = 0
    features = list(train_data)
    features = list(set(features) - set(c))
    features.remove("tag")
    features.remove("patient num")
    features.remove("test date")
    data = pd.DataFrame()
    val = pd.DataFrame()
    final_data = pd.DataFrame()
    final_data[['tag']] = train_data[['tag']]
    final_data[['patient_num']] = train_data[['patient num']]
    data[['predictions']] = cph_train[['predictions']]
    data[['duration']] = cph_train[['duration']]
    data[['test date']] = train_data[['test date']]
    val[['test date']] = val_data[['test date']]
    for feature in features:
        print(feature)
        data[feature] = train_data[feature]
        val[feature] = val_data[feature]
        predictions = run_cox_fitter(data, val)
        predictions_copy = predictions.copy()
        predictions_copy[predictions_copy <= 365] = 1
        predictions_copy[predictions_copy > 365] = 0
        y_val_copy = y_val.copy()
        y_val_copy[y_val_copy <= 365] = 1
        y_val_copy[y_val_copy > 365] = 0
        accuracy = np.mean(accuracy_score(y_val_copy, predictions_copy))
        if accuracy < prev_accuracy:
            data.drop(feature, axis=1)
            val.drop(feature, axis=1)
        else:
            final_data[feature] = train_data[feature]
        print(len(data.columns))
        prev_accuracy = accuracy
    utils.save_to_csv(final_data, "feature_selection_16_17.csv")
    print(len(final_data))


def find_k_features():
    cph = RandomForestClassifier(bootstrap=False, criterion='entropy', min_samples_leaf=1, min_samples_split=3, n_estimators=40)
    selector = RFECV(cph, 1)
    selector = selector.fit(X_train, y_train)
    print("number of features: ", selector.n_features_)
    print("ranking: ", selector.ranking_)
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
    plt.show()


#run_feature_selction()
find_k_features()


