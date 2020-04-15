import pandas as pd
import utils


# formating all dates into timestamp format
def format_date(file_name):
    data = pd.read_csv(file_name, encoding='utf8')
    columns_name = ["birth date", "aliya date", "death date"]

    for col in columns_name:
        # ignores tabs and space in the all date columns
        data[col] = data[col].str.strip()
        # turns all date columns into timestamp format
        data[col] = pd.to_datetime(data[col], format='%d.%m.%Y', errors='coerce')

    # ignores tabs and space in the "test date" column
    data["test date"] = data["test date"].str.strip()
    # turns the "test date" column into timestamp format
    data["test date"] = pd.to_datetime(data["test date"], format='%d/%m/%Y')
    return data


# turning all not numerical data into binary columns
def one_hot(data):
    one_hot_df = data[["patient num", "test date", "birth country", "population", "gender", "clinic code"]]
    # defining all columns that need to be transformed
    dummies = ["birth country", "population", "clinic code"]

    # mapping the data into its real values
    one_hot_df['population'] = one_hot_df['population'].map({1: 'Jewish', 2: 'Arabic Muslim', 3: 'Muslim', 4: 'Arabic Christian', 5: 'Christian', 6: 'Druse', 9: 'else religion'})
    one_hot_df['gender'] = one_hot_df['gender'].map({1: '0', 2: '1'})
    one_hot_df['clinic code'] = one_hot_df['clinic code'].map({10: 'clalit', 11: 'macabi', 12: 'mehuedet', 13: 'leumit', 98: 'no clinic', 99: 'unkown clinic'})

    # gender is already binary so don't need to transform
    data['gender'] = one_hot_df['gender']

    # ANA pattern transform done previously but the original wasn't deleted
    data = data.drop('ANA Pattern', 1)

    # transform all columns and setting it to data
    for field in dummies:
        embarked_dummies1 = pd.get_dummies(one_hot_df[field])
        data = data.drop(field, 1)
        data = pd.concat([data, embarked_dummies1], axis=1)
    return data


data = format_date("train.csv")
data = one_hot(data)
utils.save_to_csv(data, "train_organized.csv")
data = format_date("test.csv")
data = one_hot(data)
utils.save_to_csv(data, "test_organized.csv")
data = format_date("val.csv")
data = one_hot(data)
utils.save_to_csv(data, "val_organized.csv")