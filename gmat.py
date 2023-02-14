import numpy as np

aprime=[];bprime=[];aresult=[];rresult=[]
limit = int(input('select matrix dimension: '))
scope1 = range(0,4*limit)
scope2 = range(0,2*limit)
a = np.arange(-limit, limit+.5, .5)
b = np.arange(-limit, limit+.5, .5)
newa = np.delete(a, np.where(a==0))
newb = np.delete(b, np.where(b==0))
print('a',newa)
print('b',newb)

for axb in newa:
    if abs(axb) > .01:
        aprime.append(axb)
print('a',aprime)

for adivb in newb:
    if abs(adivb) >= 1.0001:
        bprime.append(adivb)
print('b',bprime)

aa = np.multiply.outer(aprime,bprime)
bb = np.divide.outer(aprime,bprime)
aa.tolist()
bb.tolist()
print('aa',aa), 
print('bb',bb)

for i in scope1:
    for g in scope2:
        if (((aa[i][g]) + (bb[i][g])) > 0):
            aresult.append('ACCEPT')
            print(aa[i][g] + bb[i][g])
            print('ACCEPT')
        else:
            rresult.append('REJECT: CONDITON "B" - axb + adivb !> 0')
            print(aa[i][g] + bb[i][g])
            print('REJECT: CONDITON "B"')
alenght = len(aresult)

if alenght > 0:
    print('DECISION: ')
    print('ACCEPT-  a*b > a/b in all cases')
else:
    print('REJECT: Results are mixed')