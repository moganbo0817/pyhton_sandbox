import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

parent = Path(__file__).resolve().parent
df = pd.read_csv(parent.joinpath('data/iris.csv'))
#print(df.head(2))

train2 = df.fillna(df.mean())
#print(train2)
x = train2.loc[:,:'花弁幅']
#print(x)
t = train2['種類']

sc = StandardScaler()
new = sc.fit_transform(x)

x_train,x_val,y_train,y_val = train_test_split(new,t,test_size=0.2,random_state=0)

model = LogisticRegression(random_state=0,C=0.1,multi_class='auto',solver='lbfgs')
model.fit(x_train,y_train)
score1 = model.score(x_train,y_train)
score2 = model.score(x_val,y_val)
print(score1,score2)
print(model.coef_)

x_new = [[1,2,3,4]]
print(model.predict_proba(x_new))