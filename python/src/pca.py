import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
# print(new_df.head())

new_df.columns = ['PC1','PC2']
df5 = pd.DataFrame(sc_df,columns=df4.columns)
df6 = pd.concat([df5,new_df],axis=1)

# 相関係数の計算
df_corr = df6.corr()
# print(df_corr.loc[:'very_low','PC1':]['PC1'].sort_values(ascending=False))
# print(df_corr.loc[:'very_low','PC1':]['PC2'].sort_values(ascending=False))

col = ['City','Exclusive residential']
new_df.columns = col

# new_df.plot(kind='scatter',x='City',y='Exclusive residential')
# plt.savefig(parent.joinpath('models/pca.png'))

