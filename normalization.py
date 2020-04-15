import utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import pandas as pd
import numpy


# check if its Gaussian by Shapiro method
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
    # data that shouldn't be normalized
    not_normalized_column = {"patient num", "test date",	"birth date",	"gender",
                           	"death date",	"aliya date",	"tag"}
    normalized_data = data[["patient num", "test date", "birth date", "gender",
                            "death date",	"aliya date", "tag"]]

    # initializing scalers
    standart_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    standard_features = []
    minmax_features = []

    # check which feature is gaussian and which is uniformic
    for (col_name, col_data) in data.iteritems():
        if col_name not in not_normalized_column:
            normalized = check_normalization(col_data)
            if normalized:
                standard_features.append(col_name)
            else:
                minmax_features.append(col_name)

    # scaling all the gaussian features
    for i in range(len(standard_features)):
        feature = standard_features[i]
        scaled_df = standart_scaler.fit_transform((numpy.array(data[feature])).reshape(-1, 1))
        scaled_df = pd.DataFrame(scaled_df)

        # check if the feature was really gaussian, if not doing minmax scaling
        for j in scaled_df[0]:
            if j > 1:
                scaled_df = minmax_scaler.fit_transform((numpy.array(data[feature])).reshape(-1, 1))
                break
        normalized_data[feature] = scaled_df

    # scaling all the uniformian features
    for i in range(len(minmax_features)):
        feature = minmax_features[i]
        scaled_df = minmax_scaler.fit_transform((numpy.array(data[feature])).reshape(-1, 1))
        scaled_df = pd.DataFrame(scaled_df)
        normalized_data[feature] = scaled_df

    return normalized_data


utils.save_to_csv(normalization_handler("train_organized.csv"), "normalized_train_data_16_17.csv")
utils.save_to_csv(normalization_handler("test_organized.csv"), "normalized_test_data_16_17.csv")
utils.save_to_csv(normalization_handler("val_organized.csv"), "normalized_val_data_16_17.csv")