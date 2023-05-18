def isGuusu(n):
    if num % 2 == 0 :
        s='偶数'
    else :
        s='奇数'
    return s

def greeting(i):
    if i=='こんにちは':
        s = 'ようこそ！'
    elif i =='景気は？':
        s = 'ぼちぼち！'
    elif i == 'さようなら':
        s = 'お元気で！'
    else:
        s = 'どうしました？？？'
    return s




num = int(input('整数を入力してください>>'))

print(isGuusu(num))

i = input('どしたん？')
print(greeting(i))

# if num % 2 == 0 :
#     print('偶数')
# else :
#     print('奇数')


