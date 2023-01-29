from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
import pickle
import numpy as np

def dumy(df,colum):
    job = pd.get_dummies(df[colum],drop_first=True)
    df = pd.concat([df,job],axis=1)
    df = df.drop([colum],axis=1)
    return df
    

parent = Path(__file__).resolve().parent

# CSV読み込み
# RangeIndex: 27128 entries, 0 to 27127
df = pd.read_csv(parent.joinpath('data/Bank.csv'))
#print(df.info())
#print(df.head(3))

# ダミー数変化
df = dumy(df,'marital')
df = dumy(df,'default')

#値の返還
df['duration'] = df['duration'].replace(np.nan,0)
print(df['duration'].value_counts())
print(df.head(3))

# テストデータ分割
train_val,test = train_test_split(df,test_size=0.2,random_state=0)

# 前処理
# 欠損値確認
# duration     7044
#print(df.isnull().sum())

# 欠損値補完
df = df.fillna(df.mean())
#print(df.isnull().sum())

# 特徴量候補選定
col =['age','amount','previous','campaign','duration']
x = train_val[col]
t = train_val[['y']]

def learn(x,t,depth=3):
    x_train,x_test,y_train,y_test = train_test_split(x,t,test_size=0.2,random_state=0)
    model = tree.DecisionTreeClassifier(max_depth=depth,random_state=0,class_weight='balanced')
    model.fit(x_train,y_train)
    score = model.score(X=x_train,y=y_train)
    score2 = model.score(X=x_test,y=y_test)
    return round(score,3),round(score2,3),model


for j in range(1,15):
    train_score,test_score,model = learn(x,t,depth=j)
    sentence = '訓練データの正解率{}'
    sentence2 = 'テストデータの正解率{}'
    total_stentence = '深さ{}'+sentence+sentence2
    print(total_stentence.format(j,train_score,test_score))

