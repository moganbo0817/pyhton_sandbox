listscore = range(10,0,-1)

for l in listscore:
    print(l, end ='、')

print('List off!')


def calc(n):
    return 0.8*n+20

scores =[]

for l in range(3):
    score = float(input('{}人目の得点>>'.format(l+1)))
    scores.append(score)

final_scores = list(map(calc,scores))

print(final_scores)

print(sum(final_scores)/len(final_scores))