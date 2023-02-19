import math
import numpy

ranger = range(1,9999)
fours = []
bingo = []

for i in ranger:
    if i%4 == 0:
        fours.append(str(i))

for y in fours:    
    if ((len(y) == 2) and (y[1:2] == '2')):
        bingo.append(str(y)) 
    elif ((len(y) == 3) and (y[2:3] == '2')):
        bingo.append(str(y))
    elif ((len(y) == 4) and (y[3:4] == '2')):
        bingo.append(str(y))
    elif ((len(y) == 5) and (y[4:5] == '2')):
        bingo.append(str(y))
    elif ((len(y) == 6) and (y[5:6] == '2')):
        bingo.append(str(y))
    elif ((len(y) == 7) and (y[6:7] == '2')):
        bingo.append(str(y))
    elif ((len(y) == 8) and (y[7:8] == '2')):
        bingo.append(str(y))
bingo = list((bingo))

sigma = (len(bingo))
print('There are', (sigma), 'numbers that meet this condition')


n = int(input('Enter total number of sets: '))
p = int(input('select total subsets: '))

nfac = math.factorial(n)
pfac=  math.factorial(p)

factorial = nfac//((math.factorial(n-p))*(pfac))

print(factorial)

fact = int(input('enter a digit: '))

def factorial(fact):
    prod = 1
    for i in range(0, len(fact)):
        prod = prod * fact[i]
        return prod

print(factorial(fact))


numn = int(input('select number: '))
numx = int(input('select number: '))

facti = 1
factp = 1
factq = 1

for i in range(1, numn + 1):
    facti = facti * i
for q in range(1, numx + 1):
    factq = factq * q
for p in range(1, ((numn - numx) + 1)):
    factp = factp * p

comb = ((facti)//((factp)*factq))
print(comb)




    

