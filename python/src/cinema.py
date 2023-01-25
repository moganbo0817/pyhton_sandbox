import pandas as pd
from pathlib import Path
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
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

    # 線形重回帰
    model = LinearRegression()
    model.fit(x_train, y_train)
    tmp = pd.DataFrame(model.coef_)
    tmp.index = x_train.columns
    print(tmp)
    # score = model.score(x_test,y_test)
    # print(score)

    pred = model.predict(x_test)
    # 絶対誤差
    mean = mean_absolute_error(y_pred=pred, y_true=y_test)
    print(mean)
    return model


modelPath = parent.joinpath('models/cinemamodel.pkl')

num = int(input('modelを新規作成する場合は1を入力>>'))
# モデル作成
if num == 1:
    df = createDf()
    model = createModel(df)
    with open(modelPath, 'wb') as f:
        pickle.dump(model, f)

# モデル読み込み
with open(modelPath, 'rb') as f:
    model = pickle.load(f)

x = list(map(int, input('a b c d>>').split()))
# new = [[150,700,300,0]]
new = [x]
pre = model.predict(new)
print(pre)
print('処理終了')
