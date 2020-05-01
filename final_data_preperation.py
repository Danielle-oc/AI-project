import pandas as pd
from utils import save_to_csv


def merged_final_data(imputated_data_file, binary_data_file):
    imputated_data = pd.read_csv(imputated_data_file)
    print('imputated_data')
    print(imputated_data.shape)
    binary_data = pd.read_csv(binary_data_file)
    print('binary_data')
    print(binary_data.shape)
    repeated_lables = ['tag', 'birth date', 'gender','death date', 'aliya date','AF','AR','AT','AZ','BG','BY','CZ','DE','DZ','EG','ET','GE','GR','HU','IL','IN','IQ','IR','JO','KZ','LY','MA','MN','PL','RO','RU','SY','TN','TR','UA','US','UY','UZ','YE','YU','Z3','ZA','ZZ','Arabic Christian','Arabic Muslim','Jewish','else religion','clalit','leumit','macabi','mehuedet','no clinic','unkown clinic']
    binary_data_clean = binary_data.drop(repeated_lables, axis=1)
    print('binary_data_clean')
    print(binary_data_clean.shape)
    binary_data_clean.rename(columns=lambda x: x + ' binary' if x != 'patient num' and x != 'test date' else x, inplace=True)
    merged_data = pd.merge(imputated_data, binary_data_clean, on=['patient num', 'test date' ])
    print('merged_data')
    print(merged_data.shape)
    return merged_data



train_merged_data = merged_final_data('train_imputation_binary.csv', 'train_feature_to_binary_data_16_17.csv')
save_to_csv(train_merged_data, 'train_prepared_data_16_17.csv')

test_merged_data = merged_final_data('test_imputation_binary.csv', 'test_feature_to_binary_data_16_17.csv')
save_to_csv(test_merged_data, 'test_prepared_data_16_17.csv')

val_merged_data = merged_final_data('val_imputation_binary.csv', 'val_feature_to_binary_data_16_17.csv')
save_to_csv(val_merged_data, 'val_prepared_data_16_17.csv')
