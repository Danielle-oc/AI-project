import pandas as pd
from sklearn.model_selection import train_test_split
from utils import save_to_csv_concat


# TODO: change to tagged file
data = pd.read_csv('labs_demo_tags_16_17.csv')


labels = data['tag']
features = data.drop(['tag', 'city', 'clinic'], axis=1)
X_trainVal, X_test, y_trainVal, y_test = train_test_split(features, labels, test_size=0.25)
X_train, X_val, y_train, y_val = train_test_split(X_trainVal, y_trainVal, test_size=0.2)


save_to_csv_concat(X_train, y_train, 'train.csv')
save_to_csv_concat(X_test, y_test, 'test.csv')
save_to_csv_concat(X_val, y_val, 'val.csv')

