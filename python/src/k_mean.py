import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

parent = Path(__file__).resolve().parent
df = pd.read_csv(parent.joinpath('data/Wholesale.csv'))
# print(df.head())
print(df.isnull().sum())

df = df.drop(['Channel', 'Region'], axis=1)

sc = StandardScaler()
sc_df = sc.fit_transform(df)
sd_df = pd.DataFrame(sc_df, columns=df.columns)
