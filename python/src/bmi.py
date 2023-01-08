def bmi(h,w):
    return w/(h*h)

h,w  = map(float,input('身長 体重>>').split())
print(bmi(h,w))

