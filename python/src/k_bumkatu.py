from sklearn.model_selection import KFold
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

parent = Path(__file__).resolve().parent

df = pd.read_csv(parent.joinpath('data/cinema.csv'))

df = df.fillna(df.mean())
x = df.loc[:, 'SNS1':'original']
t = df['sales']

kf = KFold(n_splits=3, shuffle=True, random_state=0)

model = LinearRegression()
result = cross_validate(model, x, t, cv=kf, scoring='r2',
                        return_train_score=True)

# print(result)
print(sum(result['test_score']/len(result['test_score'])))

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
