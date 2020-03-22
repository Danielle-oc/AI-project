import pandas as pd
import numpy as np
import utils

def labs_16_17_prep_file():
    #concatanate labs 16 and labs 17 into one table:
    labs_16 = pd.read_csv("labs_16.csv", encoding='iso8859_8', skipinitialspace=True)
    labs_17 = pd.read_csv("labs_17.csv", encoding='iso8859_8', skipinitialspace=True)
    labs_16_17 = labs_16.append(labs_17)
    labs_16_17.reset_index(drop=True)

    labs_df = pd.DataFrame(labs_16_17)
    labs_df['value'] = labs_df['value'].astype(float)
    labs_df['case num'] = labs_df['case num'].astype(float)


    services = pd.read_csv("incident_num_16_17.csv", encoding='iso8859_8', skipinitialspace=True)


    # merging labs table with services table for getting patient number:
    labs_merged = pd.merge(labs_df, services, on='case num')
    labs = labs_merged[['lab test', 'test date', 'value', 'patient num']]
    rep_labs = pd.pivot_table(labs, values='value', index=['patient num', 'test date'], columns='lab test', aggfunc=np.mean)
    rep_labs.reset_index(inplace=True)
    #print(rep_labs['ANA Pattern'])

    rep_labs['ANA Pattern'].replace({1.0: 'Speckled'})
    rep_labs['ANA Pattern'] = rep_labs['ANA Pattern'].map({1.0: 'Speckled', 2.0: 'Homogeneous', 3.0: 'centromere', 4.0: 'nucleolar', 5.0: 'midbody', 6.0: 'Nuclear dots+AMA', 7.0: 'Golgi', 8.0: 'DFS', 9.0: 'Spindle'})
    embarked_dummies1 = pd.get_dummies(rep_labs['ANA Pattern'])
    rep_labs = pd.concat([rep_labs, embarked_dummies1], axis=1)
    return rep_labs

#df_temp = rep_labs
#df_temp['anna_nan'] = rep_labs['ANA Pattern'].isna()
#print(df_temp[['anna_nan', 'ANA Pattern']][df_temp['anna_nan']!= True])
# rep_labs = labs_16_17_prep_file()
# print(rep_labs)
# # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], ['Speckled', 'Homogeneous', 'centromere', 'nucleolar', 'midbody', 'Nuclear dots+AMA', 'Golgi', 'DFS', 'Spindle']
#
# for col in rep_labs.columns:
#     print(col)



