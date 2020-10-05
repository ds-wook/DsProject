from data.datasets import train_model
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

train = train_model()
train.fillna(0, inplace=True)
print(train.columns)

clf = IsolationForest(n_estimators=50,
                      max_samples=50,
                      contamination=0.004,
                      max_features=1.0,
                      bootstrap=False,
                      n_jobs=-1,
                      random_state=2020,
                      verbose=0)

clf.fit(train.iloc[:, 1:-1])
pred = clf.predict(train.iloc[:, 1:-1])

train['anomaly'] = pred

outlier = train[train['anomaly'] == -1]
print(train['anomaly'].value_counts())
