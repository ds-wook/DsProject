# %%
from data.datasets import train_model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
train = train_model()
train.head()
# %%
kmeans = KMeans(n_clusters=150, init='k-means++', max_iter=300, random_state=0)
kmeans.fit(train.iloc[:, 1:])
# %%
print(kmeans.labels_)
# %%
train['cluster'] = kmeans.labels_
train.head()
# %%
train['cluster'].value_counts()
# %%
