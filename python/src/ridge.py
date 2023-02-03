import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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

# x_train,x_test,y_train,y_test = train_test_split(pf_x,sc_t,test_size=0.3,random_state=0)
# model = LinearRegression()
# model.fit(x_train,y_train)
# score1 = model.score(x_train,y_train)
# score2 = model.score(x_test,y_test)
# print(score1,score2)