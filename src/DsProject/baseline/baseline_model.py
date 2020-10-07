from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


def dt_model(
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dt = DecisionTreeClassifier(max_depth=3, random_state=0)
    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)
    pred_proba = dt.predict_proba(X_test)[:, 1]
    return pred, pred_proba


def rf_model(
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray) -> np.ndarray:
    rf = RandomForestClassifier(max_depth=3,
                                n_estimators=50,
                                random_state=0)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    pred_proba = rf.predict_proba(X_test)[:, 1]
    return pred, pred_proba


def xgb_model(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series) -> np.ndarray:

    xgb = XGBClassifier(
            n_jobs=-1,
            max_depth=7,
            n_estimators=1000,
            learning_rate=0.02)

    xgb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100, early_stopping_rounds=100)
    pred = xgb.predict(X_test)
    pred_proba = xgb.predict_proba(X_test)[:, 1]
    return pred, pred_proba


def lgbm_model(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series) -> np.ndarray:
    lgbm = LGBMClassifier(
            n_jobs=-1,
            max_depth=12,
            n_estimators=1000,
            learning_rate=0.02,
            num_leaves=32,
            silent=-1,
            verbose=-1)

    lgbm.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
             verbose=100, early_stopping_rounds=100)
    pred = lgbm.predict(X_test)
    pred_proba = lgbm.predict_proba(X_test)[:, 1]
    return pred, pred_proba
