from data.datasets import train_model
import pandas as pd

train = train_model()


def del_columns(
        train: pd.DataFrame) -> pd.DataFrame:
    del_features = ['Real Free %', 'Virtual free %', 'Real total(MB)',
                    'Virtual total(MB)', '%minperm', 'minfree',
                    'maxfree', '%maxclient', ' lruable pages']

    train.drop(del_features, axis=1, inplace=True)

    train.drop(['pgsin', 'pgsout', 'reclaims', 'scans', 'cycles'],
               axis=1, inplace=True)

    train = train.rename(columns={'Real free(MB)': 'Real_free',
                                  'Virtual free(MB)': 'Virtual_free'})

    return train


def change_column_name(
        train: pd.DataFrame) -> pd.DataFrame:
    columns = train.columns
    new_cols = [
        'time', 'Real_free', 'Virtual_free',
        'Process', 'FScache', 'System', 'Free',
        'Pinned', 'User', 'numperm', 'maxperm',
        'numclient', 'faults', 'pgin', 'pgout', 'week'
    ]
    rename_cols = {c1: c2 for c1, c2 in zip(columns, new_cols)}
    train.rename(columns=rename_cols, inplace=True)
 
    return train


def time_split_df(
        train: pd.DataFrame) -> pd.DataFrame:
    train['time'] = train['time'].apply(pd.to_datetime)
    train['week'] = train['time'].apply(lambda x: x.weekofyear)
    train['week'] = train['week'].apply(lambda x: 0 if x == 52 else x)
    train['dayofweek'] = train['time'].apply(lambda x: x.dayofweek)
    train['hour'] = train['time'].apply(lambda x: x.hour)
    train['day'] = train['time'].apply(lambda x: x.day)
    train['isweekend'] = train['dayofweek'].apply(lambda x: 1 if x in [5, 6]
                                                  else 0)
    return train
