# %%


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from baseline.isolate_model import train, anomaly_train


# %%


train = anomaly_train(train)
outliers = train.loc[train['anomaly'] == 0]
outlier_index = list(outliers.index)

pca = PCA(n_components=3)
scaler = StandardScaler()

X = scaler.fit_transform(train.iloc[:, 1:-1])
X_reduce = pca.fit_transform(X)


# %%


fig = plt.figure(figsize=(20, 20))
plt.title('Reduce 3 Isolation Forest')
ax = fig.add_subplot(111, projection='3d')
ax.set_zlabel('x_composite_3')

ax.scatter(X_reduce[:, 0], X_reduce[:, 1], zs=X_reduce[:, 2], s=4,
           lw=1, label='inliers', c='green')
ax.scatter(X_reduce[outlier_index, 0], X_reduce[outlier_index, 1],
           zs=X_reduce[outlier_index, 2], lw=2, s=60, marker='x', c='red',
           label='outliers')

ax.legend()
plt.show()


# %%


fig = plt.figure(figsize=(12, 8))
pca = PCA(n_components=2)
pca.fit(train.iloc[:, 1:-1])
res = pd.DataFrame(pca.transform(train.iloc[:, 1:-1]))

plt.title('Reduce 2 Isolation Forest')
plt.scatter(res[0], res[1], c='green', s=20, label='normal points')
plt.scatter(res.iloc[outlier_index, 0], res.iloc[outlier_index, 1],
            c='red', label='predicted outliers')
plt.legend()
plt.show()


# %%
