from baseline.isolate_model import train, anomaly_train
from baseline.baseline_model import xgb_model
from baseline.baseline_model import lgbm_model
from sklearn.model_selection import train_test_split
from baseline.utils import get_clf_eval
from imblearn.over_sampling import SMOTE

if __name__ == '__main__':
    train = anomaly_train(train)

    X_features = train.drop(['time', 'anomaly'], axis=1)
    y_target = train['anomaly']

    X_train, X_test, y_train, y_test = \
        train_test_split(X_features,
                         y_target,
                         random_state=2020,
                         test_size=0.2)

    # smote 기법을 활용한 over sampling
    smote = SMOTE(random_state=2020)
    X_train, y_train = smote.fit_sample(X_train, y_train)

    print('XGBoost learning!')
    xgb_pred, xgb_pred_proba = xgb_model(X_train, X_test, y_train, y_test)
    get_clf_eval(y_test, xgb_pred, xgb_pred_proba)

    print(f'\n LightGBM learning')
    lgbm_pred, lgbm_pred_proba = lgbm_model(X_train, X_test, y_train, y_test)
    get_clf_eval(y_test, lgbm_pred, lgbm_pred_proba)
