import pandas as pd
import numpy as np

def save_to_csv(y, name):
    df = pd.DataFrame(y)
    pd.DataFrame.to_csv(df, name, index=False)


def save_to_csv_concat(X, y, name):
    df = pd.concat([pd.DataFrame(y), X], axis=1)[np.append(['tag'], X.keys())]
    pd.DataFrame.to_csv(df, name, index=False)
