import pandas as pd
from sklearn.model_selection import train_test_split
from utils import save_to_csv_concat


data = pd.read_csv('labs_demo_tags_16_17.csv')
# print(data['patient num'].nunique())
row_num_per_patient = data['patient num'].value_counts().rename_axis('patient_id').reset_index(name='row_num')
# print(row_num_per_patient)
more_than_150 = []
between_100_and_149 = []
between_50_and_99 = []
between_0_and_49 = []

# print(row_num_per_patient.keys())
for i in range(len(row_num_per_patient)):
    if row_num_per_patient['row_num'][i]>149:
        more_than_150.insert(0, row_num_per_patient['patient_id'][i])
    elif row_num_per_patient['row_num'][i]>99:
        between_100_and_149.insert(0, row_num_per_patient['patient_id'][i])
    elif row_num_per_patient['row_num'][i]>49:
        between_50_and_99.insert(0, row_num_per_patient['patient_id'][i])
    else:
        between_0_and_49.insert(0, row_num_per_patient['patient_id'][i])


def split_group(patient_list, tag_patient):
    group_tag_patient = tag_patient[tag_patient['patient num'].isin(patient_list)]
    labels = group_tag_patient['tag']
    features = group_tag_patient.drop('tag', axis=1)
    X_trainVal, X_test, y_trainVal, y_test = train_test_split(features, labels, test_size=0.25)
    X_train, X_val, y_train, y_val = train_test_split(X_trainVal, y_trainVal, test_size=0.2)
    # print(list(X_train['patient num']))
    X_train = list(X_train['patient num'])
    X_val = list(X_val['patient num'])
    X_test = list(X_test['patient num'])
    return X_train, X_val, X_test



tag_patient = data[['patient num', 'tag']].drop_duplicates()
X_train_list = []
X_val_list = []
X_test_list = []

## more than 150:
X_train_temp, X_val_temp, X_test_temp = split_group(more_than_150, tag_patient)
X_train_list = X_train_list + X_train_temp
X_val_list = X_val_list + X_val_temp
X_test_list = X_test_list + X_test_temp

## 100 - 149:
X_train_temp, X_val_temp, X_test_temp = split_group(between_100_and_149, tag_patient)
X_train_list = X_train_list + X_train_temp
X_val_list = X_val_list + X_val_temp
X_test_list = X_test_list + X_test_temp

## 50 - 99:
X_train_temp, X_val_temp, X_test_temp = split_group(between_50_and_99, tag_patient)
X_train_list = X_train_list + X_train_temp
X_val_list = X_val_list + X_val_temp
X_test_list = X_test_list + X_test_temp

## 0 - 49:
X_train_temp, X_val_temp, X_test_temp = split_group(between_0_and_49, tag_patient)
X_train_list = X_train_list + X_train_temp
X_val_list = X_val_list + X_val_temp
X_test_list = X_test_list + X_test_temp

print(len(X_train_list))
print(len(X_val_list))
print(len(X_test_list))
print(len(X_train_list)+len(X_val_list)+len(X_test_list))


###split data:

# labels = data['tag']
# features = data.drop(['tag', 'city', 'clinic'], axis=1)
train = data[data['patient num'].isin(X_train_list)]
val = data[data['patient num'].isin(X_val_list)]
test = data[data['patient num'].isin(X_test_list)]

y_train = train['tag']
X_train = train.drop(['tag', 'city', 'clinic'], axis=1)
y_val = val['tag']
X_val = val.drop(['tag', 'city', 'clinic'], axis=1)
y_test = test['tag']
X_test = test.drop(['tag', 'city', 'clinic'], axis=1)

save_to_csv_concat(X_train, y_train, 'train.csv')
save_to_csv_concat(X_test, y_test, 'test.csv')
save_to_csv_concat(X_val, y_val, 'val.csv')






