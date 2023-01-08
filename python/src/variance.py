import math
#scores = [2,3,5,7,11]
scores =list(map(float,input('標準偏差を計算スペース区切りで入力>>').split()))

#print('input'+str(scores))

ave = sum(scores)/len(scores)
#print('平均'+str(ave))

hensas = list(map(lambda x: x-ave,scores))
#print('偏差'+str(hensas))

hensahensa = list(map(lambda x: x**2,hensas))
#print('偏差の2乗'+str(hensahensa))

bunsan = sum(hensahensa)/len(hensahensa)
#print('分散'+str(bunsan))

hyoujyunHensa = math.sqrt(bunsan)
print('標準偏差'+str(hyoujyunHensa))