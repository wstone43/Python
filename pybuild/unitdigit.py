ranger = range(1,201)
fours = []
twos = []  
bingo2 = []
threes =[]
bingo3=[]

for i in ranger:
    if i%4 == 0:
        fours.append(str(i))

for x in fours:
    if len(x) == 2:
        twos.append(str(x))
    elif len(x) == 3:
        threes.append(str(x))

for y in twos:    
    if y[1:2] == '2':
        bingo2.append(str(y)) 
print(bingo2)

for z in threes:    
    if z[2:3] == '2':
        bingo3.append(str(z)) 
print(bingo3)

sigma = (len(bingo2), len(bingo3))

print('There are', sum(sigma), 'numbers that meet this condition')

