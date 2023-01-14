import pandas as pd
from pathlib import Path
from sklearn import tree
import pickle

def createDf():
    # データ読み込み
    parent = Path(__file__).resolve().parent
    df = pd.read_csv(parent.joinpath('data/iris.csv'))
    #print(df.isnull().any(axis=0))
    #print('欠損')
    #print(df.isnull().sum())
    # 標準偏差
    #print(df.std())
    ## 欠損削除
    #df.dropna(how='any',axis=0, inplace=True)
    ## 平均で穴埋め
    df = df.fillna(df.mean)
    #print('欠損修正後')
    #print(df.isnull().sum())
    return df

def createModel(df):
    xcol = ['がく片長さ','がく片幅','花弁長さ','花弁幅']
    x = df[xcol]
    t = df['種類']

    return 0

df = createDf()
model = createModel()
type = df['種類'].unique()

