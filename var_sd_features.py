import pandas as pd
import numpy as np
from utils import save_to_csv_concat

def add_mean_and_var(df, new_df, col):
    mean_name = "mean " + str(col)
    var_name = "var " + str(col)
    new_df[mean_name] = (
        df.groupby('patient num')[col]
            .shift()
            .groupby(df['patient num'])
            .expanding()
            .mean()
            .values)
    new_df[var_name] = (
        df.groupby('patient num')[col]
            .shift()
            .groupby(df['patient num'])
            .expanding()
            .var()
            .values)
    new_df[var_name] = new_df[var_name].fillna(0)
    new_df[mean_name] = new_df[mean_name].fillna(df[col])
    return new_df

def create_mean_var_df(df, keys):
    sorted_data = df.sort_values(by=['patient num', 'test date'])
    X_sorted = sorted_data.drop(['tag'], axis=1)
    y_sorted = sorted_data['tag']
    new_df = pd.DataFrame()
    for key in keys:
        new_df = add_mean_and_var(X_sorted, new_df, key)
    new_X = pd.concat([X_sorted, new_df], axis=1)
    return new_X, y_sorted

val_data = pd.read_csv('val_final_data_16_17.csv')
keys = val_data.keys().drop(
        ['tag','patient num', 'test date', 'birth date', 'gender', 'aliya date', 'AF', 'AR', 'AT', 'AZ', 'BG', 'BY', 'CZ',
         'DE', 'DZ', 'EG', 'ET', 'GE', 'GR', 'HU', 'IL', 'IN', 'IQ', 'IR', 'JO', 'KZ', 'LY', 'MA', 'MN', 'PL', 'RO',
         'RU', 'SY', 'TN', 'TR', 'UA', 'US', 'UY', 'UZ', 'YE', 'YU', 'Z3', 'ZA', 'ZZ', 'Arabic Christian',
         'Arabic Muslim', 'Jewish', 'else religion', 'clalit', 'leumit', 'macabi', 'mehuedet', 'no clinic',
         'unkown clinic'])
keys_clean = keys
for key in keys:
    len = val_data[key].unique().size
    if len <= 3:
        keys_clean = keys_clean.drop([key])
print("keys:")
print(keys_clean .size)
new_val_X, new_val_y  = create_mean_var_df(val_data, keys_clean)
print("val:")
print(new_val_X)


test_data = pd.read_csv('test_final_data_16_17.csv')
new_test_X, new_test_y  = create_mean_var_df(test_data, keys_clean)
print("test:")
print(new_test_X)

train_data = pd.read_csv('train_final_data_16_17.csv')
new_train_X, new_train_y  = create_mean_var_df(train_data, keys_clean)
print("train:")
print(new_train_X)


save_to_csv_concat(new_val_X, new_val_y, "val_final_with_var_16_17.csv")
save_to_csv_concat(new_test_X, new_test_y, "test_final_with_var_16_17.csv")
save_to_csv_concat(new_train_X, new_train_y, "train_final_with_var_16_17.csv")



#
#
# val_sorted_data = val_data.sort_values(by=['patient num', 'test date'])
#
# X_val_sorted = val_sorted_data.drop(['tag'], axis=1)
# y_val_sorted = val_sorted_data['tag']
# keys = X_val_sorted.keys().drop(['patient num', 'test date', 'birth date', 'gender', 'aliya date','AF','AR','AT','AZ','BG','BY','CZ','DE','DZ','EG','ET','GE','GR','HU','IL','IN','IQ','IR','JO','KZ','LY','MA','MN','PL','RO','RU','SY','TN','TR','UA','US','UY','UZ','YE','YU','Z3','ZA','ZZ','Arabic Christian','Arabic Muslim','Jewish','else religion','clalit','leumit','macabi','mehuedet','no clinic','unkown clinic'])
# keys_clean = keys
# new_df = pd.DataFrame()
# for key in keys:
#     len = X_val_sorted[key].unique().size
#     if len <= 2:
#         keys_clean = keys_clean.drop([key])
#         print(keys_clean.size)
# for key in keys_clean:
#     new_df = add_mean_and_var(X_val_sorted,new_df, key)
# new_X_val = pd.concat([X_val_sorted, new_df], axis=1)
# save_to_csv_concat(new_X_val, y_val_sorted, "val_final_with_var_16_17.csv")
# print(new_df)
#
#
#
