import pandas as pd
from utils import save_to_csv

def convert_date_to_int(file_name):
    data = pd.read_csv(file_name)

    data['aliya date'] = pd.to_datetime(data['aliya date'])
    data['birth date'] = pd.to_datetime(data['birth date'])
    data['test date']= pd.to_datetime(data['test date'])
    data['tag'] = pd.to_datetime(data['tag'])
    data['aliya date'] = (data['aliya date'] - data['birth date']).dt.days
    data['tag'] = (data['tag'] - data['test date']).dt.days
    data['birth date'] = (data['test date'] - data['birth date']).dt.days
    data['test date'] = (data['test date'] - data['test date'].min()).dt.days

    return data



train = convert_date_to_int('train_prepared_data_16_17.csv')
save_to_csv(train, "train_final_data_16_17.csv")

test = convert_date_to_int('test_prepared_data_16_17.csv')
save_to_csv(test, "test_final_data_16_17.csv")

val = convert_date_to_int('val_prepared_data_16_17.csv')
save_to_csv(val, "val_final_data_16_17.csv")















