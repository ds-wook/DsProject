# %%
from data.datasets import train_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# %%
train = train_model()
train.head()
# %%
kmeans = KMeans(n_clusters=20, init='k-means++', max_iter=300, random_state=0)
kmeans.fit(train.iloc[:, 1:])
# %%
train['cluster'] = kmeans.labels_
# %%
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(train.iloc[:, 1:])
train['pca_x'] = pca_transformed[:, 0]
train['pca_y'] = pca_transformed[:, 1]
# %%
fig, axes = plt.subplots(figsize=(12, 10))
for i in range(kmeans.n_clusters):
    sns.scatterplot(x='pca_x', y='pca_y', data=train[train['cluster'] == i], label=i)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.title(f'{kmeans.n_clusters} cluster visualization by 2 pca components')
plt.show()

# %%
train['cluster'].value_counts()
# %%
