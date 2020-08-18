import pandas as pd
import numpy as np
from utils import save_to_csv_concat
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

train_data = pd.read_csv('train_final_data_16_17.csv')
X_train = train_data.drop(['tag'], axis=1)
y_train = train_data['tag']
y_train_copy = y_train.copy()
y_train_copy[y_train_copy <= 365] = 1
y_train_copy[y_train_copy > 365] = 0
model = KMeans(n_clusters=2)
model.fit(X_train)
cluster_train = model.predict(X_train)
cluster_train = pd.DataFrame(cluster_train, columns = ['cluster'])
cluster_train["tag"] = y_train_copy
dead_count_0 = 0
dead_size_0 = 0
dead_count_1 = 0
dead_size_1 = 0
for i in range(len(cluster_train)):
    if cluster_train["cluster"][i] == 0:
        dead_count_0 += cluster_train["tag"][i]
        dead_size_0 += 1
    else:
        dead_count_0 += cluster_train["tag"][i]
        dead_size_1 += 1
print("dead_count_0 is: " + str(dead_count_0))
print("dead_size_0 is: " + str(dead_size_0))
print("dead_count_1 is: " + str(dead_count_1))
print("dead_size_1 is: " + str(dead_size_1))






val_data = pd.read_csv('val_final_data_16_17.csv')
X_val = val_data.drop(['tag'], axis=1)
y_val = val_data['tag']
y_val_copy = y_val.copy()
y_val_copy[y_val_copy <= 365] = 1
y_val_copy[y_val_copy > 365] = 0
cluster_val = model.predict(X_val)
cluster_val = pd.DataFrame(cluster_val, columns = ['cluster'])
cluster_val["tag"] = y_val_copy
cluster_val_copy = cluster_val.copy()
cluster_val_copy[cluster_val_copy["cluster"]==0] = -1
cluster_val_copy[cluster_val_copy["cluster"]==1] = 0
cluster_val_copy[cluster_val_copy["cluster"]==-1] = 1

print("###########")
accuracy = 0
size = 0
for i in range(len(cluster_val_copy)):
    if cluster_val_copy["cluster"][i] == cluster_val["tag"][i]:
        accuracy += 1
    size += 1

print("accuracy: " + str(accuracy/size))


#
#
# # for i in range(len(cluster_val)):
# #     print(type(cluster_val["cluster"][i]))
# #
# dead_count_0 = 0
# dead_size_0 = 0
# dead_count_1 = 0
# dead_size_1 = 0
# for i in range(len(cluster_val)):
#     if cluster_val["cluster"][i] == 0:
#         dead_count_0 += cluster_val["tag"][i]
#         dead_size_0 += 1
#     else:
#         dead_count_0 += cluster_val["tag"][i]
#         dead_size_1 += 1
# print("dead_count_0 is: " + str(dead_count_0))
# print("dead_size_0 is: " + str(dead_size_0))
# print("dead_count_1 is: " + str(dead_count_1))
# print("dead_size_1 is: " + str(dead_size_1))
# ## cluster 0 - tag 1, cluster 1 - tag 0
#
#
# cluster_val_copy = cluster_val.copy()
#
# cluster_val_copy[cluster_val_copy["cluster"]==0] = -1
# cluster_val_copy[cluster_val_copy["cluster"]==1] = 0
# cluster_val_copy[cluster_val_copy["cluster"]==-1] = 1





# y_val_copy = y_val.copy()
# y_val_copy[y_val_copy <= 365] = 1
# y_val_copy[y_val_copy > 365] = -1
# print(y_val_copy)
#
# model = KMeans(n_clusters=2)
# model.fit(X_val)
# yhat = model.predict(X_val)
# print(yhat)
# print(len(yhat))


