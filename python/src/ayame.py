import pandas as pd
from pathlib import Path
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import pickle
import os
import matplotlib.pyplot as plt

parent = Path(__file__).resolve().parent

def createDf():
    # データ読み込み
    df = pd.read_csv(parent.joinpath('data/iris.csv'))
    #print(df.isnull().any(axis=0))
    #print('欠損')
    #print(df.isnull().sum())
    # 標準偏差
    #print(df.std())
    ## 欠損削除
    #df.dropna(how='any',axis=0, inplace=True)
    ## 平均で穴埋め
    df = df.fillna(df.mean())
    #print('欠損修正後')
    #print(df.isnull().sum())
    return df

def createModel(df):
    xcol = ['がく片長さ', 'がく片幅', '花弁長さ', '花弁幅']
    x = df[xcol]
    t = df['種類']    
    # 0.3をtest用、0.7を学習用
    x_train,x_test,y_train,y_test = train_test_split(x,t,test_size=0.3,random_state=0)
    model = tree.DecisionTreeClassifier(max_depth = 2,random_state=0)
    model.fit(x_train, y_train)
    score = model.score(x_test,y_test)
    print(score)
    # グラフ描画
    x_train.colums = ['gaku_naga','gaku_haba','kaben_nagasa','kaben_haba']
    plot_tree(model,feature_names=x_train.colums,filled=True,class_names=model.classes_)
    plt.savefig(parent.joinpath('models/iris.png'))
    return model

modelPath = parent.joinpath('models/ayame.pkl')

# モデルがないなら作成
if os.path.exists(modelPath) == False:
    print('model作成')
    df = createDf()
    model = createModel(df)
    with open(modelPath,'wb') as f:
        pickle.dump(model,f)
# else:
#     print('model作成')
#     df = createDf()
#     model = createModel(df)
#     with open(modelPath,'wb') as f:
#      pickle.dump(model,f)

# モデル読み込み
with open(modelPath,'rb') as f:
    model = pickle.load(f)

# 予測
#x =list(map(float,input('予想:がく片長さ がく片幅 花弁長さ 花弁幅>>').split()))
#flower = [x]
#data = pd.DataFrame(flower,columns = ['がく片長さ', 'がく片幅', '花弁長さ', '花弁幅'])
#print(model.predict(data))

print('処理終了')

