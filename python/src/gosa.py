from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.metrics import mean_absolute_error
import math
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd
from pathlib import Path

parent = Path(__file__).resolve().parent

df = pd.read_csv(parent.joinpath('data/cinema.csv'))
df = df.fillna(df.mean())

x = df.loc[:, 'SNS1':'original']
t = df['sales']

model = LinearRegression()
model.fit(x, t)

pred = model.predict(x)

mse = mean_squared_error(pred, t)

# print(math.sqrt(mse))


# yosoku = [2, 3, 5, 7, 11, 13]
# target = [3, 5, 8, 11, 16, 19]

# mse = mean_squared_error(yosoku, target)
# print('rmse:{}'.format(math.sqrt(mse)))
# print('mae:{}'.format(mean_absolute_error(yosoku, target)))

# print('はずれ値の代入')

# yosoku = [2, 3, 5, 7, 11, 46]
# target = [3, 5, 8, 11, 16, 23]

# mse = mean_squared_error(yosoku, target)
# print('rmse:{}'.format(math.sqrt(mse)))
# print('mae:{}'.format(mean_absolute_error(yosoku, target)))

df = pd.read_csv(parent.joinpath('data/Survived.csv'))
df = df.fillna(df.mean())

x = df[['Pclass', 'Age']]
t = df['Survived']

model = tree.DecisionTreeClassifier(max_depth=2, random_state=0)
model.fit(x, t)


pred = model.predict(x)
out_put = classification_report(y_pred=pred, y_true=t, output_dict=True)
print(pd.DataFrame(out_put))
