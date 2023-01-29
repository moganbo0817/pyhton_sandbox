from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pickle
import numpy as np

def dumy(df,colum):
    job = pd.get_dummies(df[colum],drop_first=True)
    df = pd.concat([df,job],axis=1)
    df = df.drop([colum],axis=1)
    return df
    
def printBar(df,colum):
    marital = df.groupby(colum).mean()
    bar = marital['y'].plot(kind='bar')
    plt.savefig(parent.joinpath('models/bank_bar_'+colum+'.png'))

def printSca(df,colum):
    df.plot(kind='scatter',x=colum,y='y')
    plt.savefig(parent.joinpath('models/bank_sca_'+colum+'.png'))

def printAveBar(df,colum):
    amount = df.groupby('y').mean()[colum]
    bar = amount.plot(kind='bar')
    plt.xlabel(colum)
    plt.savefig(parent.joinpath('models/bank_ave_bar_'+colum+'.png'))


parent = Path(__file__).resolve().parent

# CSV読み込み
# RangeIndex: 27128 entries, 0 to 27127
df = pd.read_csv(parent.joinpath('data/Bank.csv'))
#print(df.info())
#print(df.head(3))

print(df['y'].value_counts())

#　グラフ描画
#printBar(df,'default')
#printBar(df,'marital')
#printBar(df,'amount')

#　各要素平均
#printAveBar(df,'campaign')

print(pd.pivot_table(df,index="housing",columns="loan",values="duration"))

# ダミー数変化
df = dumy(df,'marital')
df = dumy(df,'default')

#値の返還
df['duration'] = df['duration'].replace(np.nan,0)
#print(df['duration'].value_counts())
#print(df.head(3))

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

# def learn(x,t,depth=3):
#     x_train,x_test,y_train,y_test = train_test_split(x,t,test_size=0.2,random_state=0)
#     model = tree.DecisionTreeClassifier(max_depth=depth,random_state=0,class_weight='balanced')
#     model.fit(x_train,y_train)
#     score = model.score(X=x_train,y=y_train)
#     score2 = model.score(X=x_test,y=y_test)
#     return round(score,3),round(score2,3),model


# for j in range(1,15):
#     train_score,test_score,model = learn(x,t,depth=j)
#     sentence = '訓練データの正解率{}'
#     sentence2 = 'テストデータの正解率{}'
#     total_stentence = '深さ{}'+sentence+sentence2
#     print(total_stentence.format(j,train_score,test_score))


test = test.fillna(train_val.mean())
x_test= test[col]
t_test = test[['y']]
model = tree.DecisionTreeClassifier(max_depth=12,random_state=0,class_weight='balanced')
model.fit(x_test,t_test)
print(model.score(x_test,t_test))

# 後でもう1回もどってこよう
