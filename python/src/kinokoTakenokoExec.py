import pandas as pd
from pathlib import Path
from sklearn import tree
import pickle

# モデル読み込み
parent = Path(__file__).resolve().parent
with open(parent.joinpath('models/KinokoTakenoko.pkl'),'rb') as f:
    model = pickle.load(f)

# 予測
x =list(map(int,input('予想：身長 体重 年代>>').split()))
taro = [x]
data = pd.DataFrame(taro,columns = ['身長','体重','年代'])
print(model.predict(data))