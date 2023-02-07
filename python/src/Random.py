from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

parent = Path(__file__).resolve().parent
df = pd.read_csv(parent.joinpath('data/Survived.csv'))
# print(df.head(2))

jo1 = df['Pclass'] == 1
jo2 = df['Survived'] == 0
jo3 = df['Age'].isnull()
df.loc[(jo1) & (jo2) & (jo3), 'Age'] = 43

jo2 = df['Survived'] == 1
df.loc[(jo1) & (jo2) & (jo3), 'Age'] = 35

jo1 = df['Pclass'] == 2
jo2 = df['Survived'] == 0
jo3 = df['Age'].isnull()
df.loc[(jo1) & (jo2) & (jo3), 'Age'] = 26

jo2 = df['Survived'] == 1
df.loc[(jo1) & (jo2) & (jo3), 'Age'] = 20

jo1 = df['Pclass'] == 3
jo2 = df['Survived'] == 0
jo3 = df['Age'].isnull()
df.loc[(jo1) & (jo2) & (jo3), 'Age'] = 43

jo2 = df['Survived'] == 1
df.loc[(jo1) & (jo2) & (jo3), 'Age'] = 35

col = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

x = df[col]
t = df['Survived']

dummy = pd.get_dummies(df['Sex'], drop_first=True)
x = pd.concat([x, dummy], axis=1)
# print(x.head)

x_train, x_test, y_train, y_test = train_test_split(
    x, t, test_size=0.2, random_state=0)

model = RandomForestClassifier(n_estimators=200, random_state=0)
model.fit(x_train, y_train)

score1 = model.score(x_train, y_train)
score2 = model.score(x_test, y_test)
print(score1, score2)

model2 = tree.DecisionTreeClassifier(random_state=0)
model2.fit(x_train, y_train)

score1 = model2.score(x_train, y_train)
score2 = model2.score(x_test, y_test)
print(score1, score2)

importance = model.feature_importances_
im = pd.Series(importance, index=x_train.columns)
print(im)
