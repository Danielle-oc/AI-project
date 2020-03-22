from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from utilities import find_corr
from sklearn.impute import SimpleImputer
import numpy as np
from compare_df_by_alogrithms import compare_df
from utilities import save_to_csv


# fetch the cleansed data and normalize the Age_group feature between 0-1
train_data = pd.read_csv('train_cleansing.csv')
train_data['Age_group'] = train_data['Age_group'].map({1: 0.3, 2: 0.6,  3: 1})
X_train = train_data.drop(['Vote'], axis=1)
y_train = train_data['Vote']

test_data = pd.read_csv('test_cleansing.csv')
test_data['Age_group'] = test_data['Age_group'].map({1: 0.3, 2: 0.6,  3: 1})
X_test = test_data.drop(['Vote'], axis=1)
y_test = test_data['Vote']

val_data = pd.read_csv('val_cleansing.csv')
val_data['Age_group'] = val_data['Age_group'].map({1: 0.3, 2: 0.6,  3: 1})
X_val = val_data.drop(['Vote'], axis=1)
y_val = val_data['Vote']

# created for the correlation test
X_train_no_cat = X_train.drop(['Most_Important_Issue', 'Occupation', 'Main_transportation'], axis=1)


def get_corrs(data):
    d_size = len(data.keys())
    corrs = [[] for i in range (d_size)]
    for i in range(d_size):
        for j in range(d_size):
            f1 = data.keys()[i]
            f2 = data.keys()[j]
            corrs[i].append(find_corr(data, f1, f2))
    return corrs


def check_corr(data, old_corrs):
    new_corrs = get_corrs(data)
    for i in range(len(new_corrs)):
        for j in range(len(new_corrs)):
            if i < j:
                if abs(old_corrs[i][j]) > 0.5:
                    if abs(new_corrs[i][j]) < (abs(old_corrs[i][j]) - 0.01) or \
                            abs(new_corrs[i][j]) > (abs(old_corrs[i][j]) + 0.01):
                        print("features: ", data.keys()[i], ", ", data.keys()[j], ", new corr: ", new_corrs[i][j],
                              " old corr: ", old_corrs[i][j])


def normalize(train_data, target_data):
    # origin_corrs = get_corrs(train_data) # used for test
    new_data = target_data.copy()

    # manipulate data
    minmax_ftrs = ['Occupation_Satisfaction', 'Yearly_IncomeK', 'Overall_happiness_score',
                   'Garden_sqr_meter_per_person_in_residancy_area', 'Yearly_ExpensesK', 'Last_school_grades',
                   'Number_of_differnt_parties_voted_for', 'Number_of_valued_Kneset_members',
                   'Num_of_kids_born_last_10_years']

    zscore_ftrs = ['Avg_monthly_expense_when_under_age_21', 'AVG_lottary_expanses',
                   'Avg_monthly_expense_on_pets_or_plants', 'Avg_environmental_importance', 'Avg_Residancy_Altitude',
                   'Avg_government_satisfaction', 'Avg_Satisfaction_with_previous_vote', 'Avg_monthly_household_cost',
                   'Phone_minutes_10_years', 'Avg_size_per_room', 'Weighted_education_rank',
                   'Avg_monthly_income_all_years', 'Political_interest_Total_Score', 'Avg_education_importance']

    # minmax scaling
    minmax_scaler = MinMaxScaler()
    for i in range(len(minmax_ftrs)):
        feature = minmax_ftrs[i]
        scaled_df = minmax_scaler.fit_transform((np.array(train_data[feature])).reshape(-1, 1))
        scaled_df = pd.DataFrame(scaled_df)
        new_data[feature] = scaled_df

    # standard scaling
    standart_scaler = StandardScaler()
    for i in range(len(zscore_ftrs)):
        feature = zscore_ftrs[i]
        scaled_df = standart_scaler.fit_transform((np.array(train_data[feature])).reshape(-1, 1))
        scaled_df = pd.DataFrame(scaled_df)
        new_data[feature] = scaled_df

    # division by 100 scaling
    new_data['%Time_invested_in_work'] /= 100
    new_data['%_satisfaction_financial_policy'] /= 100

    # check_corr(new_data, origin_corrs) # assert the correlation haven't been destroyed nor improved
    return new_data


X_train = normalize(X_train, X_train)
X_test = normalize(X_train, X_test)
X_val = normalize(X_train, X_val)


# save raw
save_to_csv(X_train, y_train, 'train_normalizing.csv')
save_to_csv(X_test, y_test, 'test_normalizing.csv')
save_to_csv(X_val, y_val, 'val_normalizing.csv')


# tests the algorithms on new and old dfs:
#
# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#
# new_data = normalize(X_train_no_cat, X_train_no_cat)
# imp = imp.fit(new_data)
# new_data_imp = pd.DataFrame(imp.transform(new_data))
#
# old_data = X_train_no_cat
# imp = imp.fit(old_data)
# old_data_imp = pd.DataFrame(imp.transform(old_data))
#
# # new_data = pd.concat([old_data_imp, train_data['Vote']], axis=1)
# # old_data = pd.concat([new_data_imp, train_data['Vote']], axis=1)
# new_data_imp['Vote'] = y_train
# old_data_imp['Vote'] = y_train
#
# print(train_data['Vote'])
# compare_df(new_data_imp, old_data_imp)