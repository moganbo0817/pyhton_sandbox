import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
#print(df2)

# df2[['temp','hum']].plot(kind='line')
# plt.savefig(parent.joinpath('models/bike_temp.png'))
# df2['temp'].plot(kind='hist')
# df2['hum'].plot(kind='hist',alpha=0.5)
# plt.savefig(parent.joinpath('models/bike_hist.png'))

#df2['atemp'].loc[220:240].plot(kind='line')
#plt.savefig(parent.joinpath('models/bike_atemp.png'))

df2['atemp'] = df2['atemp'].astype(float)
df2['atemp'] = df2['atemp'].interpolate()

df2['atemp'].loc[220:240].plot(kind='line')
#plt.savefig(parent.joinpath('models/bike_atemp.png'))

iris_df = pd.read_csv(parent.joinpath('data/iris.csv'))
non_df = iris_df.dropna()
x = non_df.loc[:,'がく片幅':'花弁幅']
t = non_df['がく片長さ']
model = LinearRegression()
model.fit(x,t)

condition = iris_df['がく片長さ'].isnull()
non_data = iris_df.loc[condition]

x = non_data.locc[:,'がく片幅':'花弁幅']
pred = model.predict(x)

iris_df.loc[condition,'がく片長さ'] = pred
