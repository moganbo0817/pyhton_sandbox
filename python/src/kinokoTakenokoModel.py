import pandas as pd
from pathlib import Path
from sklearn import tree
import pickle

# データ読み込み
parent = Path(__file__).resolve().parent
df = pd.read_csv(parent.joinpath('data/KvsT.csv'))

# 特徴量
xcol =['身長','体重','年代']
x = df[xcol]

# 正解データ
t = df['派閥']

# モデル
model = tree.DecisionTreeClassifier(random_state=0)

# 学習
model.fit(x,t)

# 保存
with open(parent.joinpath('models/KinokoTakenoko.pkl'),'wb') as f:
    pickle.dump(model,f)

# 正解率
#exac = model.score(x,t)
#print('正解率:'+str(exac))

# 予測
#taro = [[170,70,20],[158,48,20]]
#data = pd.DataFrame(taro,columns = ['身長','体重','年代'])
#print(model.predict(data))