from pathlib import Path
import pandas as pd

parent = Path(__file__).resolve().parent

# CSV読み込み
# RangeIndex: 27128 entries, 0 to 27127
df = pd.read_csv(parent.joinpath('data/Bank.csv'))
#print(df.info())
#print(df.head(3))

# ダミー数変化
df = dumy(df)
print(df.head(3))

# テストデータ分割


# 前処理

# 欠損値確認
# duration     7044
#print(df.isnull().sum())

# 欠損値補完
df = df.fillna(df.mean())
#print(df.isnull().sum())

# 特徴量候補選定

def dumy(df):
    return df
