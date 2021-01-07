from baseline.isolate_model import train, anomaly_train
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    train = anomaly_train(train)

    X_features = train.drop(["time", "anomaly"], axis=1)
    y_target = train["anomaly"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_target, random_state=2020, test_size=0.2
    )

    # smote 기법을 활용한 over sampling
    smote = SMOTE(random_state=2020)
    X_train, y_train = smote.fit_sample(X_train, y_train)

    xgb_clf = XGBClassifier(learning_rate=0.01, n_jobs=-1, random_state=94)
    xgb_clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])

    feature_importances = pd.DataFrame()
    feature_importances["feature"] = train.iloc[:, 1:-1].columns
    feature_importances["importance"] = xgb_clf.feature_importances_
    feature_importances = feature_importances.sort_values(
        by="importance", ascending=False
    )
    plt.figure(figsize=(12, 8))
    plt.title("XGB importance top-10")
    sns.barplot(x="importance", y="feature", data=feature_importances[:10])
    plt.show()
