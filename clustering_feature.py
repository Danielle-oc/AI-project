import pandas as pd
import numpy as np
from utils import save_to_csv_concat
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

train_data = pd.read_csv('train_final_data_16_17.csv')
X_train = train_data.drop(['tag'], axis=1)
y_train = train_data['tag']

###check best cluster num:
# sse = []
# list_k = list(range(1, 10))
#
# for k in list_k:
#     km = KMeans(n_clusters=k)
#     km.fit(X_train)
#     sse.append(km.inertia_)
#
# plt.figure(figsize=(6, 6))
# plt.plot(list_k, sse, '-o')
# plt.xlabel(r'Number of clusters *k*')
# plt.ylabel('Sum of squared distance')
# plt.show()

model = KMeans(n_clusters=2)
model.fit(X_train)
cluster_train = model.predict(X_train)
print(len(cluster_train))
X_train["cluster"] = cluster_train
save_to_csv_concat(X_train, y_train, "train_final_cluster_k3_16_17.csv")




val_data = pd.read_csv('val_final_data_16_17.csv')
X_val = val_data.drop(['tag'], axis=1)
y_val = val_data['tag']
model = KMeans(n_clusters=2)
model.fit(X_val)
cluster_val = model.predict(X_val)
print(len(cluster_val))
X_val["cluster"] = cluster_val
save_to_csv_concat(X_val, y_val, "val_final_cluster_k3_16_17.csv")


test_data = pd.read_csv('test_final_data_16_17.csv')
X_test = test_data.drop(['tag'], axis=1)
y_test = test_data['tag']
model = KMeans(n_clusters=2)
model.fit(X_test)
cluster_test = model.predict(X_test)
print(len(cluster_test))
X_test["cluster"] = cluster_test
save_to_csv_concat(X_test, y_test, "test_final_cluster_k3_16_17.csv")








