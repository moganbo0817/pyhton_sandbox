import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from pathlib import Path

parent = Path(__file__).resolve().parent
df = pd.read_csv(parent.joinpath('data/cinema.csv'))

df = df.fillna(df.mean())
x = df.loc[:,'SNS1':'original']
t = df['sales']

x_train,x_test,y_train,y_test = train_test_split(x,t,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=0,n_estimators=100)

model.fit(x_train,y_train)
print(model.score(x_test,y_test))

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

base = DecisionTreeRegressor(random_state=0,max_depth=3)

model = AdaBoostRegressor(random_state=0,n_estimators=100,base_estimator=base)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))