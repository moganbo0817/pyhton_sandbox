list = []

nationalLang = int(input('国語の点数>>'))
list.append(nationalLang)

math = int(input('数学の点数>>'))
list.append(math)

eng = int(input('英語の点数>>'))
list.append(eng)

print(list)

print(sum(list))