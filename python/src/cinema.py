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
    df = pd.read_csv(parent.joinpath('data/cinema.csv'))
    df = df.fillna(df.mean())

    # グラフ描画
    # for name in df.columns:
    #     if name == 'cinema_id' or name == 'sales':
    #         continue
    #     df.plot(kind='scatter', x=name, y='sales')
    #     plt.savefig(parent.joinpath('models/cinema_scatter_'+name+'.png'))

    df.shape
    no = df[(df['SNS2'] > 1000) & (df['sales'] < 8500)].index
    df = df.drop(no, axis=0)
    return df


def createModel(df):
    x = df.loc[:, 'SNS1':'original']
    t = df['sales']
    x_train, x_test, y_train, y_test = train_test_split(
        x, t, test_size=0.2, random_state=0)

    return 0


modelPath = parent.joinpath('models/irismodel.pkl')
df = createDf()
model = createModel(df)

print('処理終了')
