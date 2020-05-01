import pandas as pd
import utils

# Turn all columns to binary that shows if the lab test was done or not
def get_feature_in_binary(file_name):
    data = pd.read_csv(file_name, encoding='utf8')
    # find all the columns that have null values (these are only test columns and dates columns)
    null_list = data.columns[data.isna().any()].tolist()
    no_binary = ['death date', 'aliya date']
    # remove date columns
    list_to_binary = list(set(null_list) - set(no_binary))
    for feature in list_to_binary:
        # if the cell has value than 1 else 0
        data[feature] = data[feature].apply(lambda x: 1 if not pd.isna(x) else 0)
    return data

data = get_feature_in_binary("normalized_train_data_16_17.csv")
utils.save_to_csv(data, "train_feature_to_binary_data_16_17.csv")
data = get_feature_in_binary("normalized_test_data_16_17.csv")
utils.save_to_csv(data, "test_feature_to_binary_data_16_17.csv")
data = get_feature_in_binary("normalized_val_data_16_17.csv")
utils.save_to_csv(data, "val_feature_to_binary_data_16_17.csv")
