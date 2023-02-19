import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

parent = Path(__file__).resolve().parent

df = pd.read_csv(parent.joinpath('data/Boston.csv'))
# print(df.head())

# 欠損値の穴埋め（平均）
df2 = df.fillna(df.mean())

# ダミー変化
dummy = pd.get_dummies(df2['CRIME'], drop_first=True)
df3 = df2.join(dummy)
df3 = df3.drop(['CRIME'], axis=1)
# print(df.head())

# 標準化
# flotに変換
df4 = df3.astype('float')
# 標準化
sc = StandardScaler()
sc_df = sc.fit_transform(df4)

model = PCA(n_components=2, whiten=True)
model.fit(sc_df)

# print(model.components_[0])
# print('-----')
# print(model.components_[1])

new = model.transform(sc_df)
new_df = pd.DataFrame(new)
print(new_df.head())
