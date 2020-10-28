# %%


from data.data_load import mem, memnew, memuse
from data.data_load import page  # 데이터 로드

import pandas as pd
import numpy as np
import seaborn as sns  # 데이터 시각화
import matplotlib.pyplot as plt  # 기본 시각화
import statsmodels.api as sm   # 가설검정
from statsmodels.formula.api import ols  # ols model
import warnings
warnings.filterwarnings('ignore')



# %% [markdown]


'''
# 문제가 될 수 있는 가능성.
```
1.   특정 프로세스가 CPU를 많이 차지해서 문제
2.   특정 프로세스가 memory를 많이 차지해서 문제
3.   Paging space (Swap) 가상의 메모리 공간을 디스크 일부-> 메모리-디스크 간에 스왑이 일어나면 문제가 생김 ->
     Nmon에서 paging space in/out을 봐야함
4.   Disk io가 높은데 특정 디스크만 많이 써서 문제
5.   Network의 대역폭이 커서, 원래 처리할 수 있는 데이터보다 많이 처리할 때
6.   Process별로 resource 사용량을 저장하고, 어떤 CPU가 어떤 메모리를 사용했는지.
7.   평소보다 많이 썻네? 어떤놈이 많이 썻는지 보기 위해서 process를 봄.
8.   CPU, MEMORY는 사용량이 높은데 점유율이 높은 process가 없을 때는 ? process가 많이 뜬거임.
```
'''


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


train['time'] = train['time'].apply(pd.to_datetime)
train['week'] = train['time'].apply(lambda x: x.weekofyear)
train['week'] = train['week'].apply(lambda x: 0 if x == 52 else x)


# %% [markdown]
'''
# feature 내용
```
1. Real Free % : 사용 가능한 총 RAM에 대한 사용 가능한 실제 RAM의 백분율
2. Virtual free % : 할당 된 총 페이징 공간에 대한 사용 가능한 페이징 공간의 백분율
3. Real free(MB) : 사용 가능한 RAM 공간 (MB)
4. Vistual free(MB) : 사용 가능한 페이징 공간 (MB)
5. Real total(MB) : 총 페이징 공간 크기 (MB)
6. Virtual total(MB) : 특정 시간에 공유 메모리 풀에서 공유 메모리 파티션에 할당되는 실제 메모리 (MB)
7. Process% : 시스템에서 사용 가능한 총 실제 메모리와 비교하여 프로세스에서 사용하는 실제 메모리의 백분율
8. FScache% : 실제 메모리와 비교하여 파일 시스템 캐시에서 사용하는 실제 메모리의 백분율
9. System% : 실제 메모리와 비교하여 시스템 세그먼트에서 사용하는 메모리 비율
10. Free% : 총 RAM 대비 사용 가능한 RAM 비율
11. Pinned% : 사용 가능한 메모리에 대한 고정 된 메모리의 백분율입니다. 이 고정 된 메모리는 고정된 작업 세그먼트,
              고정된 영구 세그먼트,고정된 클라이먼트 세그먼트의 합계입니다.
12. User% : 비 시스템 페이지는 사용자 세그먼트로 분류됩니다. user%실제 메모리에 대한 사용자 세그먼트의 비율
13. %numperm : 실제 메모리 크기에 대한 계산할 수없는 페이지의 프레임 비율
14. %minperm : 페이지 스틸러가 복제 비율에 관계없이 파일 또는 계산 페이지를 훔칠 때까지의 시간을 지정합니다. 페이지 스틸러는
               비계산 페이지를 캐시하기 위해 최소 메모리 양을 목표
15. %maxperm : 페이지 탈취 알고리즘이 파일 페이지만
               탈취하는 데 걸리는 시간을 지정
16. minfree : VMM (Virtual Memory Manager)이 여유 목록을 다시 채우기 위해 페이지를 훔치기 시작하는 여유
목록의 최소 프레임 수를 지정
17. maxfree : 페이지 탈취가 중지 된 후 사용 가능한 목록의 프레임 수를 지정
18. %numclient : 사용 가능한 총 실제 메모리에 대한 클라이언트 프레임 수의 백분율(?)
19. %maxclient : 사용 가능한 총 실제 메모리에 대한 클라이언트 프레임 수의 백분율(?)
20. lruable pages : LRU (Least Recent Used) 알고리즘으로 처리 할 수있는 페이지
21. faults :	초당 페이지 부재 수
22. pgin(page in) :	초당 페이지 인 작업 수
23. pgout(page out) :	초당 페이지 아웃 작업 수
```
'''
# %%


del_features = [
    'Real Free %', 'Virtual free %', 'Real total(MB)',
    'Virtual total(MB)', '%minperm', 'minfree',
    'maxfree', '%maxclient', ' lruable pages',
]
train.drop(del_features, axis=1, inplace=True)

train.drop(['pgsin', 'pgsout', 'reclaims', 'scans', 'cycles'],
           axis=1, inplace=True)

train = train.rename(columns={'Real free(MB)': 'Real_free',
                              'Virtual free(MB)': 'Virtual_free'})
train.head()


# %%


columns = train.columns
new_cols = [
    'time', 'Real_free', 'Virtual_free',
    'Process', 'FScache', 'System', 'Free',
    'Pinned', 'User', 'numperm', 'maxperm',
    'numclient', 'faults', 'pgin', 'pgout', 'week'
]
rename_cols = {c1: c2 for c1, c2 in zip(columns, new_cols)}
train.rename(columns=rename_cols, inplace=True)
train.head()


# %%


train_grouped = train.groupby('week', as_index=False)
for col in new_cols[1:-1]:
    f, ax = plt.subplots(figsize=(20, 8))
    sns.boxplot(x='week', y=col, data=train, ax=ax)
    sns.pointplot(x='week', y=col, data=train_grouped[col].mean())
    plt.title(col + ' about week(box)')
    plt.show()
    results = ols(f'{col}~week', data=train).fit()
    print(results.summary())
    anova_table = sm.stats.anova_lm(results)
    print(anova_table)


# %%
train['dayofweek'] = train['time'].apply(lambda x: x.dayofweek)
# %%
train_grouped = train.groupby('dayofweek', as_index=False)
for col in new_cols[1:-1]:
    f, ax = plt.subplots(figsize=(20, 8))
    sns.boxplot(x='dayofweek', y=col, data=train, ax=ax)
    sns.pointplot(x='dayofweek', y=col, data=train_grouped[col].mean())
    plt.title(col + ' about weekday')
    plt.xticks(np.arange(0, 7), labels=['Mon', 'Tue', 'Wen',
                                        'Thr', 'Fri', 'Sat', 'Sun'])
    plt.show()
    results = ols(f'{col}~week', data=train).fit()
    print(results.summary())
    anova_table = sm.stats.anova_lm(results)
    print(anova_table)
# %%
train['hour'] = train['time'].apply(lambda x: x.hour)
# %%
for col in new_cols[1:-1]:
    f, ax = plt.subplots(figsize=(20, 8))
    sns.pointplot(x='hour', y=col, data=train, ax=ax)
    plt.title(col + ' about hour')
    plt.show()
# %%
train['day'] = train['time'].apply(lambda x: x.day)
# %%
for col in new_cols[1:-1]:
    f, ax = plt.subplots(figsize=(20, 8))
    sns.pointplot(x='day', y=col, data=train, ax=ax)
    plt.title(col + ' about day')
    plt.show()
# %%
train['hour_of_day'] = train['time'].apply(lambda x: f'{x.dayofweek}-{x.hour}')
# %%
for col in new_cols[1:-1]:
    f, ax = plt.subplots(figsize=(32, 8))
    sns.lineplot(x='hour_of_day', y=col, data=train, ax=ax)
    plt.title(col + ' about hour of day')
    plt.xticks(rotation=75)
    plt.show()
# %%
for col in new_cols[1:-1]:
    f, ax = plt.subplots(figsize=(20, 8))
    sns.lineplot(x='hour', y=col, hue='dayofweek', data=train, ax=ax, ci=None)
    plt.title(col + 'about hour of day')
    plt.show()
# %%
for col in new_cols[1:-1]:
    f, ax = plt.subplots(figsize=(20, 8))
    sns.boxplot(x='hour', y=col, hue='dayofweek', data=train, ax=ax)
    plt.title(col + ' about hour of day')
    plt.show()
# %%
train['isweekend'] = train['dayofweek'].apply(lambda x: 1 if x in [5, 6]
                                              else 0)
# %%
df = train[(train['week'] != 0) & (train['week'] != 5)]
train_grouped = df.groupby(['week', 'isweekend'], as_index=False)
for col in new_cols[1:-1]:
    f, ax = plt.subplots(figsize=(20, 8))
    sns.boxplot(x='week', y=col, hue='isweekend', data=df,
                palette=sns.xkcd_palette(['sky blue', 'pink']), ax=ax)
    sns.pointplot(x='week', y=col, hue='isweekend', data=train_grouped.mean(),
                  palette=sns.xkcd_palette(['sky blue', 'hot pink']), ax=ax)
    plt.title(col + ' about week of weekend')
    plt.show()
    results = ols(f'{col}~week + isweekend', data=train).fit()
    print(results.summary())
    anova_table = sm.stats.anova_lm(results)
    print(anova_table)

# %%
grouped = train.groupby(['hour', 'isweekend'])
for col in new_cols[1:-1]:
    f, ax = plt.subplots(figsize=(20, 8))

    sns.boxplot(x='hour', y=col, hue='isweekend',
                palette=sns.xkcd_palette(['pastel blue', 'pastel red']),
                data=train, ax=ax)

    sns.pointplot(x='hour', y=col, hue='isweekend',
                  data=grouped[col].mean().reset_index(), ax=ax)

    plt.title(col + ' about hour of weekend')
    plt.show()
    results = ols(f'{col}~hour + isweekend', data=train).fit()
    print(results.summary())
    anova_table = sm.stats.anova_lm(results)
    print(anova_table)
# %%
fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(train.corr(), fmt='.2f', annot=True)
plt.show()

# %%


from sklearn.ensemble import IsolationForest
if_clf = IsolationForest(n_estimators=50,
                          max_samples=50,
                          contamination=0.004,
                          max_features=1.0,
                          bootstrap=False,
                          n_jobs=-1,
                          random_state=2020,
                          verbose=0)

if_clf.fit(train.iloc[:, 1:])
feature_importance = pd.DataFrame()
feature_importance['feature'] = train.iloc[:, 1:].columns
feature_importance.head()
# %%
train['anomaly'] = if_clf.predict(train.iloc[:, 1:])
train['anomaly'] = train['anomaly'].replace(-1, 0)
train['anomaly'].value_counts()
# %%
from lightgbm import LGBMClassifier
lgbm_clf = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.02,
    num_leaves=32,
    max_depth=12)
lgbm_clf.fit(train.iloc[:, 1:-1], train['anomaly'])
feature_importance['importance'] = lgbm_clf.feature_importances_
feature_importance = feature_importance.sort_values(by='importance', ascending=False)
feature_importance.head()
# %%
plt.figure(figsize=(20, 12))
plt.title('Isolate Forest by LightGBM')
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.show()
# %%
