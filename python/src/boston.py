import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

parent = Path(__file__).resolve().parent
df = pd.read_csv(parent.joinpath('data/Boston.csv'))

crime_count = df['CRIME'].value_counts()

crime = pd.get_dummies(df['CRIME'], drop_first=True)
df = pd.concat([df, crime], axis=1)
df = df.drop(['CRIME'], axis=1)

train_val, test = train_test_split(df, test_size=0.2, random_state=0)

# print(train_val.isnull().sum())
train_val = train_val.fillna(train_val.mean())

# 行削除 CTL SHIFT K
# for name in train_val.columns:
#     train_val.plot(kind='scatter',x=name,y='PRICE')
#     plt.savefig(parent.joinpath('models/boston.scatter_'+name+'.png'))
out_line1 = train_val[(train_val['RM'] < 6) & (train_val['PRICE'] > 40)].index
out_line2 = train_val[(train_val['PTRATIO'] > 18) &
                      (train_val['PRICE'] > 40)].index
# print(out_line1,out_line2)
train_val = train_val.drop([76], axis=0)

col = ['INDUS', 'NOX', 'RM', 'PTRATIO', 'LSTAT', 'PRICE']
train_val = train_val[col]
# print(train_val.head())
train_cor = train_val.corr()['PRICE']
abs_cor = train_cor.map(abs).sort_values(ascending=False)
# print(abs_cor)

col = ['RM', 'LSTAT', 'PTRATIO']
x = train_val[col]
t = train_val[['PRICE']]

x_train, x_val, y_train, y_val = train_test_split(
    x, t, test_size=0.2, random_state=0)

sc_model_x = StandardScaler()
sc_model_x.fit(x_train)

sc_x = sc_model_x.transform(x_train)
# print(sc_x)
# tmp_df = pd.DataFrame(sc_x,columns=x_train.columns)
# print(tmp_df.std())

sc_model_y = StandardScaler()
sc_model_y.fit(y_train)

sc_y = sc_model_y.transform(y_train)

model = LinearRegression()
model.fit(sc_x, sc_y)
# print(model.score(x_val,y_val))

sc_x_val = sc_model_x.transform(x_val)
sc_y_val = sc_model_y.transform(y_val)

# print(model.score(sc_x_val,sc_y_val))


def learn(x, t):
    x_train, x_val, y_train, y_val = train_test_split(
        x, t, test_size=0.2, random_state=0)

    sc_model_x = StandardScaler()
    sc_model_x.fit(x_train)
    sc_x_train = sc_model_x.transform(x_train)

    sc_model_y = StandardScaler()
    sc_model_y.fit(y_train)
    sc_y_train = sc_model_y.transform(y_train)

    model = LinearRegression()
    model.fit(sc_x_train, sc_y_train)

    sc_x_val = sc_model_x.transform(x_val)
    sc_y_val = sc_model_y.transform(y_val)

    train_score = model.score(sc_x_train, sc_y_train)
    val_score = model.score(sc_x_val, sc_y_val)

    return train_score, val_score


x = train_val.loc[:, ['RM', 'LSTAT', 'PTRATIO']]
t = train_val[['PRICE']]

s1, s2 = learn(x, t)
print(s1, s2)
