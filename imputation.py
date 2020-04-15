import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn import mixture
import random
from utils import save_to_csv_concat


train_data = pd.read_csv('normalized_data_16_17.csv')
X_train = train_data.drop(['tag'], axis=1)
y_train = train_data['tag']

# test_data = pd.read_csv('test_normalizing.csv')
# X_test = test_data.drop(['tag'], axis=1)
# y_test = test_data['tag']
#
# val_data = pd.read_csv('val_normalizing.csv')
# X_val = val_data.drop(['tag'], axis=1)
# y_val = val_data['tag']



X_train_no_cat = X_train.drop(['birth country','patient num','birth date', 'death date','aliya date', 'test date','ANA Pattern'], axis=1)

# checking for corelation between two features
def fill_data_missing_by_interpolate(corrs):
    for f1 in X_train_no_cat.keys():
        null_count = X_train_no_cat[f1].isna().sum()
        zero_count = X_train_no_cat[X_train_no_cat[f1] == 0][f1].count()
        if null_count < 5000 and zero_count < 5000:
            curr_corrs = corrs[f1]
            corrs[f1][f1] = 0

            j = np.nanargmax(abs(curr_corrs))
            max_corr_abs = curr_corrs[j]
            null_count = X_train_no_cat[f1].isna().sum()
            if abs(max_corr_abs) >= 0.9:
                f2 = X_train_no_cat.keys()[j]
                print('f1 is ' + f1 + ' best match ' + f2 + ' with score ' + str(max_corr_abs))

def check_corellation():
    data = X_train_no_cat
    corrs = data.corr(method='pearson', min_periods=1)
    fill_data_missing_by_interpolate(corrs)



# find minumum value for each field

for f1 in X_train_no_cat.keys():
    print(f1 + ' minimum value is ' + str(X_train_no_cat[f1].min()))
