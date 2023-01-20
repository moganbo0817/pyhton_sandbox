import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import matplotlib.pyplot as plt
import pickle

parent = Path(__file__).resolve().parent
df = pd.read_csv(parent.joinpath('data/Survived.csv'))
count = df['Survived'].value_counts()
# print(df.isnull().sum())
# print(df.shape)

#df['Age'] = df['Age'].fillna(df['Age'].mean())
#df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# print(df['Age'].mean())
# print(df['Age'].median())

#print(df.groupby('Pclass').mean()['Age'])
#piv = pd.pivot_table(df,index='Survived',columns='Pclass',values='Age')
#print(piv)
#piv = pd.pivot_table(df,index='Survived',columns='Pclass',values='Age',aggfunc=max)
#print(piv)

is_null = df['Age'].isnull()

#条件指定してAge列を抜き出して平均値で穴埋め
df.loc[(df['Pclass']==1) & (df['Survived']==0) & (is_null),'Age'] = 43
df.loc[(df['Pclass']==1) & (df['Survived']==1) & (is_null),'Age'] = 35

df.loc[(df['Pclass']==2) & (df['Survived']==0) & (is_null),'Age'] = 33
df.loc[(df['Pclass']==2) & (df['Survived']==1) & (is_null),'Age'] = 25

df.loc[(df['Pclass']==3) & (df['Survived']==0) & (is_null),'Age'] = 26
df.loc[(df['Pclass']==3) & (df['Survived']==1) & (is_null),'Age'] = 20

sex = df.groupby('Sex').mean()
#sex['Survived'].plot(kind='bar')
#plt.savefig(parent.joinpath('models/sex.png'))
male = pd.get_dummies(df['Sex'],drop_first=True)
#print(male)

emb = pd.get_dummies(df['Embarked'],drop_first=True)
#print(emb)

col = ['Pclass','Age','SibSp','Parch','Fare']
x = df[col]
t = df['Survived']

x = pd.concat([x,male],axis=1)

#x_train,x_test,y_train,y_test = train_test_split(x,t,test_size=0.2,random_state=0)
#model = tree.DecisionTreeClassifier(max_depth=5,random_state=0,class_weight='balanced')
#model.fit(x_train,y_train)
#score = model.score(X=x_test,y=y_test)

def learn(x,t,depth=3):
    x_train,x_test,y_train,y_test = train_test_split(x,t,test_size=0.2,random_state=0)
    model = tree.DecisionTreeClassifier(max_depth=depth,random_state=0,class_weight='balanced')
    model.fit(x_train,y_train)
    score = model.score(X=x_train, y=y_train)
    score2 = model.score(X=x_test,y=y_test)
    
    return round(score,3),round(score2,3),model

# for j in range(1,15):
#     train_score,test_score,model = learn(x,t,depth=j)
#     sentence = '訓練データの正解率{}'
#     sentence2 = 'テストデータの正解率{}'
#     total_senetence= '深さ{}'+sentence+sentence2
#     print(total_senetence.format(j,train_score,test_score))

train_score,test_score,model = learn(x,t,depth=5)
modelPath = parent.joinpath('models/survived.pkl')

with open(modelPath,'wb') as f:
    pickle.dump(model,f)
