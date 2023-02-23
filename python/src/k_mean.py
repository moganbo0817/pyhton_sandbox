import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
print(sc_df.head(2))
