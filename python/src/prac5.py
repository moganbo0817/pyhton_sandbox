def isLeapYear(y):
    if y % 4 ==0:
        if y%100 == 0:
            if y%400 == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False

#year = int(input('年を入力>>'))

#print(isLeapYear(year))

## ng ng ok