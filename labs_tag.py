import pandas as pd
import labs
import utils


#merge the labs csv we created with the lab tests as column and only numerical data with demographic data
def demo_merge():
    rep_labs = labs.labs_16_17_prep_file()
    demo = pd.read_csv("demographic_16_17.csv", encoding='iso8859_8')
    labs_merged = pd.merge(rep_labs, demo, on='patient num')
    return labs_merged


#finds the date of the last test and returns it plus 3 years
def get_expected_death_date(i, patient_dates):
    patient_num = patient_dates["patient num"][i]
    max_date = patient_dates["test date"][i]
    while patient_dates["patient num"][i] == patient_num:
        if(patient_dates["test date"][i] > max_date):
            max_date = patient_dates["test date"][i]
        i = i + 1
        if i == len(patient_dates["patient num"]):
            break
    tag_date = max_date.replace(year=max_date.year + 3)
    return str(tag_date.day) + "." + str(tag_date.month) + "." + str(tag_date.year)

#adds a columns of expected death date to the the file with the lab tests and the demographic data
def add_tag(data):

    patient_dates = data[["patient num", "test date", "death date"]]
    #ignores tabs and space in the "test date" column
    patient_dates["test date"] = patient_dates["test date"].str.strip()
    #turns the "test date" column into timestamp format
    patient_dates["test date"] = pd.to_datetime(patient_dates["test date"], format='%d/%m/%Y')
    #sorting the DF by patient num and sub-sorting by test date
    patient_dates = patient_dates.sort_values(["patient num", "test date"], ascending=[True, True])

    #dictionary with patient num as key and the eaxpected death date as value
    patient_tag = {}

    for num in patient_dates["patient num"]:
        patient_tag[num] = 0

    for i in range(len(patient_dates["patient num"])):
        if patient_tag[patient_dates["patient num"][i]] != 0:
            continue
        #if the patient died his tag will be his real death date
        if patient_dates["death date"][i] != '00.00.0000':
            patient_tag[patient_dates["patient num"][i]] = patient_dates["death date"][i]
        else:
            patient_tag[patient_dates["patient num"][i]] = get_expected_death_date(i, patient_dates)
    #adding tag column to the DF
    data['tag'] = 0
    #setting each patient with his expected death date
    for i in range(len(data["patient num"])):
        data['tag'][i] = patient_tag[data["patient num"][i]]
    utils.save_to_csv(data, "labs_demo_tags_16_17.csv")
    return data


def get_all_data_tagged():
    data = demo_merge()
    return add_tag(data)


get_all_data_tagged()