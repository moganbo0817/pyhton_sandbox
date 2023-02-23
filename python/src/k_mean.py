import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

parent = Path(__file__).resolve().parent
df = pd.read_csv(parent.joinpath('data/Wholesale.csv'))
# print(df.head())
# print(df.isnull().sum())

df = df.drop(['Channel', 'Region'], axis=1)

sc = StandardScaler()
sc_df = sc.fit_transform(df)
sc_df = pd.DataFrame(sc_df, columns=df.columns)

model = KMeans(n_clusters = 3, random_state = 0)
model.fit(sc_df)
model.labels_
sc_df['cluster'] = model.labels_
# print(sc_df.head(2))
# print(sc_df.groupby('cluster').mean())
cluster_mean = sc_df.groupby('cluster').mean()
# cluster_mean.plot(kind='bar')
# plt.savefig(parent.joinpath('models/kmean.png'))

sse_list = []

for n in range(2,31):
    model = KMeans(n_clusters= n, random_state=0)
    model.fit(sc_df)
    sse = model.inertia_
    sse_list.append(sse)

# print(sse_list)
se = pd.Series(sse_list)
num = range(2,31)
# se.index = num
se.plot(kind='line')
plt.savefig(parent.joinpath('models/k_mean_line.png'))

model = KMeans(n_clusters=5,random_state=0)
model.fit(sc_df)
sc_df['cluster'] = model.labels_
sc_df.to_csv(parent.joinpath('data/clusterd_Wholeasale.csv'), index= False)