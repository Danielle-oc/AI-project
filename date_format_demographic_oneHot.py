import pandas as pd
import utils


def format_date(file_name):
    data = pd.read_csv(file_name, encoding='utf8')
    columns_name = ["birth date", "aliya date", "death date"]

    #ignores tabs and space in the "test date" column
    for col in columns_name:
        #turns the "test date" column into timestamp format
        data[col] = data[col].str.strip()
        data[col] = pd.to_datetime(data[col], format='%d.%m.%Y', errors='coerce')

    data["test date"] = data["test date"].str.strip()
    #turns the "test date" column into timestamp format
    data["test date"] = pd.to_datetime(data["test date"], format='%d/%m/%Y')
    return data


def oneHot(data):
    one_hot_df = data[["patient num", "test date", "birth country", "population", "gender", "clinic code"]]
    dummies = ["birth country", "population", "clinic code"]
    one_hot_df['population'] = one_hot_df['population'].map({1: 'Jewish', 2: 'Arabic Muslim', 3: 'Muslim', 4: 'Arabic Christian', 5: 'Christian', 6: 'Druse', 9: 'else religion'})
    one_hot_df['gender'] = one_hot_df['gender'].map({1: '0', 2: '1'})
    one_hot_df['clinic code'] = one_hot_df['clinic code'].map({10: 'clalit', 11: 'macabi', 12: 'mehuedet', 13: 'leumit', 98: 'no clinic', 99: 'unkown clinic'})
    data['gender'] = one_hot_df['gender']
    data = data.drop('ANA Pattern', 1)
    for field in dummies:
        data[field] = one_hot_df[field]
        embarked_dummies1 = pd.get_dummies(one_hot_df[field])
        data = data.drop(field, 1)
        data = pd.concat([data, embarked_dummies1], axis=1)
    return data


data = format_date("train.csv")
data = oneHot(data)
utils.save_to_csv(data, "train_organized.csv")
data = format_date("test.csv")
data = oneHot(data)
utils.save_to_csv(data, "test_organized.csv")
data = format_date("val.csv")
data = oneHot(data)
utils.save_to_csv(data, "val_organized.csv")