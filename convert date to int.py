import pandas as pd
from utils import save_to_csv

def convert_date_to_int(file_name):
    data = pd.read_csv(file_name)

    data['test date']= pd.to_datetime(data['test date'])
    data['test date'] = (data['test date'] - data['test date'].min()).dt.days
    data['tag'] = pd.to_datetime(data['tag'])
    data['tag'] = (data['tag'] - data['tag'].min()).dt.days
    data['birth date'] = pd.to_datetime(data['birth date'])
    data['birth date'] = (data['birth date'] - data['birth date'].min()).dt.days
    data['aliya date'] = pd.to_datetime(data['aliya date'])
    data['aliya date'] = (data['aliya date'] - data['aliya date'].min()).dt.days

    return data



train = convert_date_to_int('train_prepared_data_16_17.csv')
save_to_csv(train, "train_final_data_16_17.csv")

test = convert_date_to_int('test_prepared_data_16_17.csv')
save_to_csv(test, "test_final_data_16_17.csv")

val = convert_date_to_int('val_prepared_data_16_17.csv')
save_to_csv(val, "val_final_data_16_17.csv")














#
#
# print(type(data['test date'][1]))
# data['test date']= pd.to_datetime(data['test date'])
# print(type(data['test date'][1]))
#
#
# print(data['test date'].min())
#
# print('#########')
# print( type( (data['test date'] - data['test date'].min()).dt.days)  )
# data['test date'] = (data['test date'] - data['test date'].min()).dt.days
# data['te'] = (data['test date'] - data['test date'].min()).dt.days
#
#
