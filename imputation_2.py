import pandas as pd
from utils import save_to_csv_concat

## delete death date and imputate aliya date
def clean_data(X_data):
    X_data_clean = X_data.drop(['death date'], axis=1)
    for i in range(len(X_data_clean['aliya date'])):
        if type(X_data_clean['aliya date'][i]) != str:
            X_data_clean['aliya date'][i] = X_data_clean['birth date'][i]
    return X_data_clean


def create_non_binary_list(X_data):
    binary_list = []
    for f1 in X_data.keys():
        flag = True
        for i in range(len(X_data[f1])):
            if X_data[f1][i] == X_data[f1][i]:
                if X_data[f1][i] != 1 and X_data[f1][i] != 0:
                    flag = False
        if not flag:
            binary_list.append(f1)
    return binary_list

def create_binary_list(X_data):
    binary_list = []
    for f1 in X_data.keys():
        flag = True
        for i in range(len(X_data[f1])):
            if X_data[f1][i] == X_data[f1][i]:
                if X_data[f1][i] != 1 and X_data[f1][i] != 0:
                    flag = False
        if flag:
            binary_list.append(f1)
    return binary_list


def imputation_binary(file_name):
    data = pd.read_csv(file_name)
    X_data = data.drop(['tag'], axis=1)
    y_data = data['tag']
    X_data_clean = clean_data(X_data)
    no_bin_list = create_non_binary_list(X_data_clean)

    null_list = X_data_clean.columns[X_data_clean.isna().any()].tolist()
    str_list = ['aliya date']

    list_to_binary = list(set(null_list) - set(no_bin_list) - set(str_list))
    for feature in list_to_binary:
        X_data_clean[feature] = X_data_clean[feature].apply(lambda x: -1 if pd.isna(x) else x)
    return X_data_clean, y_data


def imputation_non_binary(X_data_clean, y_data):
    bin_list = create_binary_list(X_data_clean)

    null_list = X_data_clean.columns[X_data_clean.isna().any()].tolist()
    str_list = ['aliya date']

    list_to_binary = list(set(null_list) - set(bin_list) - set(str_list))
    for feature in list_to_binary:
        X_data_clean[feature] = X_data_clean[feature].apply(lambda x: X_data_clean[feature].mean() if pd.isna(x) else x)
    return X_data_clean, y_data





X_train, y_train = imputation_binary('normalized_train_data_16_17.csv')
X_train, y_train = imputation_non_binary(X_train, y_train)
save_to_csv_concat(X_train, y_train, 'train_imputation_binary.csv')

X_test, y_test = imputation_binary('normalized_test_data_16_17.csv')
X_test, y_test = imputation_non_binary(X_test, y_test)
save_to_csv_concat(X_test, y_test, 'test_imputation_binary.csv')

X_val, y_val = imputation_binary('normalized_val_data_16_17.csv')
X_val, y_val = imputation_non_binary(X_val, y_val)
save_to_csv_concat(X_val, y_val, 'val_imputation_binary.csv')

