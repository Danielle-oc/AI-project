import pandas as pd
from sklearn.model_selection import train_test_split
from utilities import save_to_csv

data = pd.read_csv('ElectionsData.csv')

# map features to numeric values
data['Vote'] = data['Vote'].map({'Khakis': 0, 'Oranges': 1, 'Purples': 2, 'Turquoises': 3, 'Yellows': 4, 'Blues': 5,
                                'Whites': 6, 'Violets': 7, 'Reds': 8, 'Pinks': 9, 'Greys': 10, 'Greens': 11,
                                 'Browns': 12}).astype(int)
data['Looking_at_poles_results'] = data['Looking_at_poles_results'].map({'Yes': 1, 'No': -1})
data['Married'] = data['Married'].map({'Yes': 1, 'No': -1})
data['Gender'] = data['Gender'].map({'Female': 1, 'Male': -1})
data['Voting_Time'] = data['Voting_Time'].map({'By_16:00': 1, 'After_16:00': -1})
data['Will_vote_only_large_party'] = data['Will_vote_only_large_party'].map({'Maybe': 0, 'Yes': 1, 'No': -1})
data['Age_group'] = data['Age_group'].map({'Below_30': 1, '30-45': 2,  '45_and_up': 3})
data['Financial_agenda_matters'] = data['Financial_agenda_matters'].map({'Yes': 1, 'No': -1})

# todo delete and move to imputation
# # one hot
# embarked_dummies1 = pd.get_dummies(data['Most_Important_Issue'])
# data = pd.concat([data, embarked_dummies1], axis=1)
# embarked_dummies2 = pd.get_dummies(data['Main_transportation'])
# data = pd.concat([data, embarked_dummies2], axis=1)
# embarked_dummies3 = pd.get_dummies(data['Occupation'])
# data = pd.concat([data, embarked_dummies3], axis=1)
# data = data.drop(['Most_Important_Issue', 'Main_transportation', 'Occupation'], axis=1)

# split
labels = data['Vote']
features = data.drop(['Vote'], axis=1)
X_trainVal, X_test, y_trainVal, y_test = train_test_split(features, labels, test_size=0.25)
X_train, X_val, y_train, y_val = train_test_split(X_trainVal, y_trainVal, test_size=0.2)

# save raw
save_to_csv(X_train, y_train, 'train.csv')
save_to_csv(X_test, y_test, 'test.csv')
save_to_csv(X_val, y_val, 'val.csv')
