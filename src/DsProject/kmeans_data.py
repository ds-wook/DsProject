# %%
from data.data_load import mem, memnew, memuse
from data.data_load import page
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
mem.rename(columns={'Memory pcordb02': 'time'}, inplace=True)
memnew.rename(columns={'Memory New pcordb02': 'time'}, inplace=True)
memuse.rename(columns={'Memory Use pcordb02': 'time'}, inplace=True)
page.rename(columns={'Paging pcordb02': 'time'}, inplace=True)

# %%
train = mem.merge(memnew, on=['time'], how='outer')
for dataset in [memuse, page]:
    train = train.merge(dataset, on=['time'], how='outer')
# %%
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, random_state=0)
kmeans.fit(train.iloc[:, 1:])
# %%
print(kmeans.labels_)

# %%
train['cluster'] = kmeans.labels_
train.head()
# %%
train['cluster'].value_counts()
# %%
X, y = make_blobs(n_samples=200, n_features=2, centers=3,
                  cluster_std=0.8, random_state=0)
print(X.shape, y.shape)
# %%
cluster_df = pd.DataFrame(data=X, columns=['ftr1', 'ftr2'])
cluster_df['target'] = y
cluster_df.head()
# %%
target_list = np.unique(y)
for target in target_list:
    target_data = cluster_df[cluster_df['target'] == target]
    sns.scatterplot(x='ftr1', y='ftr2', data=target_data)
plt.show()
# %%
