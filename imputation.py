import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn import mixture
import random
from utils import save_to_csv_concat


train_data = pd.read_csv('normalized_test_data_16_17.csv')
X_train = train_data.drop(['tag'], axis=1)
y_train = train_data['tag']

# test_data = pd.read_csv('test_normalizing.csv')
# X_test = test_data.drop(['tag'], axis=1)
# y_test = test_data['tag']
#
# val_data = pd.read_csv('val_normalizing.csv')
# X_val = val_data.drop(['tag'], axis=1)
# y_val = val_data['tag']



# X_train_no_cat = X_train.drop(['birth country','patient num','birth date', 'death date','aliya date', 'test date','ANA Pattern'], axis=1)

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
X_train_clean = X_train.drop(['death date'], axis=1)
for i in range(len(X_train_clean['aliya date'])):
    if type(X_train_clean['aliya date'][i])!= str:
        X_train_clean['aliya date'][i] = X_train_clean['birth date'][i]



# for i in range(len(X_train_clean['aliya date'])):
#     print(type(X_train_clean['aliya date'][i])!= str, X_train_clean['aliya date'][i])




# for f1 in X_train_clean.keys():
#     flag  = True
#     for i in range(len(X_train_clean[f1])):
#         if flag and X_train_clean[f1][i] != X_train_clean[f1][i]:
#                 if X_train_clean[f1].min() != 0:
#                     print(f1 + ' minimum value is ' + str(X_train_clean[f1].min()))
#                 flag = False


bin_dict = dict.fromkeys(X_train_clean.keys(), False)
print(bin_dict)

for f1 in X_train_clean.keys():
    flag = True
    for i in range(len(X_train_clean[f1])):
        if X_train_clean[f1][i] == X_train_clean[f1][i]:
            if X_train_clean[f1][i] != 1 and X_train_clean[f1][i] != 0:
                flag = False
    bin_dict[f1] = flag
print(bin_dict)



X_train_clean_binary = X_train_clean

for f1 in X_train_clean_binary.keys():
    print('f1 is')
    print(f1)
    if bin_dict[f1]:
        for i in range(len(X_train_clean_binary[f1])):
            if X_train_clean_binary[f1][i] != X_train_clean_binary[f1][i]:
                X_train_clean_binary[f1][i] = -1;

save_to_csv_concat(X_train_clean_binary, y_train, 'imputation_binary')

# for i in range(len(X_train_clean[f1])):
#     mean = X_train_clean[f1].mean()
#     if X_train_clean[f1][i] != X_train_clean[f1][i]:
#         if bin_dict[f1]:
#             X_train_clean[f1][i] = -1;
#             print('binary')
#         else:
#             X_train_clean[f1][i] = mean
#             print('non-binary')
#
# save_to_csv_concat(X_train_clean, y_train, 'imputation_temp')



########## all change


# for f1 in X_train_clean.keys():
#     print('f1 is')
#     print(f1)
#     for i in range(len(X_train_clean[f1])):
#         mean = X_train_clean[f1].mean()
#         if X_train_clean[f1][i] != X_train_clean[f1][i]:
#             if bin_dict[f1]:
#                 X_train_clean[f1][i] = -1;
#                 print('binary')
#             else:
#                 X_train_clean[f1][i] = mean
#                 print('non-binary')
#
# save_to_csv_concat(X_train_clean, y_train, imputation_temp)

# for f1 in X_train_no_cat.keys():
#     if type(X_train[f1][1]) =
