import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

parent = Path(__file__).resolve().parent
df = pd.read_csv(parent.joinpath('data/Survived.csv'))
count = df['Survived'].value_counts()
print(df.isnull().sum())
print(df.shape)

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

col = ['Pclass','Age','SibSp','Parch','Fare']
x = df[col]
t = df['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,t,test_size=0.2,random_state=0)
print(x_train.shape)