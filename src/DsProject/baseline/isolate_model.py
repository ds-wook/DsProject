from data.datasets import train_model
from baseline.fea_eng import del_columns
from baseline.fea_eng import change_column_name
from baseline.fea_eng import time_split_df
import pandas as pd
from sklearn.ensemble import IsolationForest

train = train_model()
train.fillna(0, inplace=True)

train = del_columns(train)
train = change_column_name(train)
train = time_split_df(train)


def anomaly_train(train: pd.DataFrame) -> pd.DataFrame:
    clf = IsolationForest(
        n_estimators=50,
        max_samples=50,
        contamination=0.004,
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=2020,
        verbose=0,
    )

    clf.fit(train.iloc[:, 1:-1])
    pred = clf.predict(train.iloc[:, 1:-1])
    train["anomaly"] = pred
    train["anomaly"] = train["anomaly"].replace(-1, 0)

    return train
