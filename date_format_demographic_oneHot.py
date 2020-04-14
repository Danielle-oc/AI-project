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
        for i in range(len(data["birth date"])):
            if data[col][i] in (None, ""):
                data[col][i] = '00/00/0000'
    data["test date"] = data["test date"].str.strip()
    #turns the "test date" column into timestamp format
    data["test date"] = pd.to_datetime(data["test date"], format='%d/%m/%Y')
    file_name = file_name.split('.')[0]
    utils.save_to_csv(data, file_name + "_organized.csv")


format_date("train.csv")