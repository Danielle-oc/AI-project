import utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import pandas as pd
import numpy


def check_normalization(feature_set):
    feature_set = numpy.asarray(feature_set)
    k2, p = stats.shapiro(feature_set)
    alpha = 0.005
    if p >= alpha:
        return True
    else:
        return False


def normalization_handler(file_name):
    data = pd.read_csv(file_name, encoding='utf8')
    not_normalized_column = {"patient num", "test date", "clinic code", "clinic",	"birth date", "birth country",	"gender",
                           "city",	"death date",	"aliya date",	"population",	"tag", "ANA Pattern"}
    normalized_data = data[["patient num", "test date", "clinic code", "birth date", "birth country",	"gender",
                            "death date",	"aliya date",	"population",	"tag", "ANA Pattern"]]
    standart_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    standard_features = []
    minmax_features = []
    for (col_name, col_data) in data.iteritems():
        if col_name not in not_normalized_column:
            normalized = check_normalization(col_data)
            if normalized:
                standard_features.append(col_name)
            else:
                minmax_features.append(col_name)

    for i in range(len(standard_features)):
        feature = standard_features[i]
        scaled_df = standart_scaler.fit_transform((numpy.array(data[feature])).reshape(-1, 1))
        scaled_df = pd.DataFrame(scaled_df)
        normalized_data[feature] = scaled_df

    for i in range(len(minmax_features)):
        feature = minmax_features[i]
        scaled_df = minmax_scaler.fit_transform((numpy.array(data[feature])).reshape(-1, 1))
        scaled_df = pd.DataFrame(scaled_df)
        normalized_data[feature] = scaled_df
    #normalized_data[standard_features] = standart_scaler.fit_transform(data[standard_features])
    #normalized_data[minmax_features] = minmax_scaler.fit_transform(data[minmax_features])
    return normalized_data


utils.save_to_csv(normalization_handler("train.csv"), "normalized_train_data_16_17.csv")
utils.save_to_csv(normalization_handler("test.csv"), "normalized_test_data_16_17.csv")
utils.save_to_csv(normalization_handler("val.csv"), "normalized_val_data_16_17.csv")