import pandas as pd
from sklearn import tree
from pathlib import Path

parent = Path(__file__).resolve().parent

df = pd.read_csv(parent.joinpath('data/Kvst.csv'))
x = df.loc[:,'体重':'年代']
t = df['派閥']

model = tree.DecisionTreeClassifier(max_depth=1,random_state=0)
model.fit(x,t)
data = [[65,20]]

print(model.predict(data))
print(model.predict_proba(data))