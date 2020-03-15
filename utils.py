import pandas as pd


def save_to_csv(y, name):
    df = pd.DataFrame(y)
    pd.DataFrame.to_csv(df, name, index=False)
