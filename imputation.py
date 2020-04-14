import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn import mixture
import random
from utils import save_to_csv_concat


train_data = pd.read_csv('train_normalizing.csv')
X_train = train_data.drop(['Vote'], axis=1)
y_train = train_data['Vote']

test_data = pd.read_csv('test_normalizing.csv')
X_test = test_data.drop(['Vote'], axis=1)
y_test = test_data['Vote']

val_data = pd.read_csv('val_normalizing.csv')
X_val = val_data.drop(['Vote'], axis=1)
y_val = val_data['Vote']
