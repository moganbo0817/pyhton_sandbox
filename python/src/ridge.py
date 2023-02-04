import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

parent = Path(__file__).resolve().parent
df = pd.read_csv(parent.joinpath('data/Boston.csv'))
df = df.fillna(df.mean())
df = df.drop([76],axis=0)

t = df[['PRICE']]
x = df.loc[:,['RM','PTRATIO','LSTAT']]

sc = StandardScaler()
sc_x = sc.fit_transform(x)
sc2 = StandardScaler()
sc_t = sc2.fit_transform(t)

pf = PolynomialFeatures(degree=2,include_bias=False)
pf_x = pf.fit_transform(sc_x)
#print(pf.get_feature_names_out())

x_train,x_test,y_train,y_test = train_test_split(pf_x,sc_t,test_size=0.3,random_state=0)
# model = LinearRegression()
# model.fit(x_train,y_train)
# score1 = model.score(x_train,y_train)
# score2 = model.score(x_test,y_test)
# print(score1,score2)

# ridgeMode = Ridge(alpha=10)
# ridgeMode.fit(x_train,y_train)
# score1 = ridgeMode.score(x_train,y_train)
# score2 = ridgeMode.score(x_test,y_test)
# print(score1,score2)

maxScore = 0
maxIndex = 0

# for i in range(1,2001):
#     num = i/100
#     rideModel = Ridge(random_state=0,alpha=num)
#     rideModel.fit(x_train,y_train)
#     result = rideModel.score(x_test,y_test)
#     # print(result)
#     if result > maxScore:
#         maxScore = result
#         maxIndex = num

# print(maxScore,maxIndex)

# model = LinearRegression()
# model.fit(x_train,y_train)

# rideModel = Ridge(random_state=0,alpha=17.62)
# rideModel.fit(x_train,y_train)

# print(sum(abs(model.coef_)[0]))
# print(sum(abs(rideModel.coef_)[0]))

# from sklearn.linear_model import Lasso

# model = Lasso(alpha=0.1)
# model.fit(x_train,y_train)
# score1 = model.score(x_train,y_train)
# score2 = model.score(x_test,y_test)
# # print(score1,score2)
# weight = model.coef_
# ws = pd.Series(weight,index=pf.get_feature_names_out())
# print(ws)

df = pd.read_csv(parent.joinpath('data/Boston.csv'))
df = df.fillna(df.mean())
x = df.loc[:,'ZN':'LSTAT']
print(x.shape)
t = df['PRICE']
print(t.shape)

x_train,x_test,y_train,y_test = train_test_split(x,t,test_size=0.3,random_state=0)
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=10,random_state=0)
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print(score)

s = pd.Series(model.feature_importances_,index=x.columns)
print(s)