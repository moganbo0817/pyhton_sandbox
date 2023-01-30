import pandas as pd
from pathlib import Path

parent = Path(__file__).resolve().parent
df = pd.read_csv(parent.joinpath('data/bike.tsv'), sep='\t')
#print(df.head(2))

weather = pd.read_csv(parent.joinpath('data/weather.csv'), encoding='shift-jis')
#print(weather)

temp = pd.read_json(parent.joinpath('data/temp.json'))
#print(temp.T)

df2 = df.merge(weather,how='inner', on='weather_id')
#print(df2)
#print(df2.groupby('weather').mean()['cnt'])

#print(temp.T.loc[199:201])
#print(df2[df2['dteday']=='2011-07-20'])

df2 = df2.merge(temp.T,how='left',on='dteday')
print(df2)