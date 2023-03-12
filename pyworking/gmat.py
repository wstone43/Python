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

print('aa',aa)
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



# ///////////////////////////////////////OTHER METHODS///////////////////////////////////////////////////////




# import numpy as np

# limit = float(input('select matrix dimension: '))
# results=[];results2=[];results3=[]

# x = np.arange(-limit,0,.1)


# for ranger in x:
#     if ranger**2>=0:
#         results.append(ranger)
# if len(results) == len(x):
#     print('A: YES')
# else:
#     print('A: NO')

# for ranger in x:
#     if ranger-(2*ranger)>0:
#         results2.append(ranger)
# if len(results2) == len(x):
#     print('B: YES')
# else:
#     print('B: NO')

# for ranger in x:
#     if ranger**2 + ranger**3<0:
#         results3.append(ranger)
# if len(results3) == len(x):
#     print('C: YES')
# else:
#     print('C: NO')







# import numpy as np

# limit = int(input('select matria dimension: '))
# results=[];results2=[];results3=[]

# a = np.arange(-limit,0,.1)

# def formula(test1,test2,test3):
#     for ranger in test1:
#         if ranger>=0:
#             results.append(ranger)
#     if len(results) == len(a):
#         print('A: YES')
#     else:
#         print('A: NO')

#     for ranger in test2:
#         if ranger>0:
#             results2.append(ranger)
#     if len(results2) == len(a):
#         print('B: YES')
#     else:
#         print('B: NO')

#     for ranger in test3:
#         if ranger<0:
#             results3.append(ranger)
#     if len(results3) == len(a):
#         print('C: YES')
#     else:
#         print('C: NO')

# formula((a**2),(a-(2*a)),(a**2+a**3))

# print(len(results))
# print(len(a))












import numpy as np

lim = int(input('Select matrix dimension: '))
res1=[];res2=[];dump1=[];dump2=[]

x = np.arange(-lim,lim,.1)
y = np.arange(-lim,lim,.1)

def jumper(disco,rifle):
    for ranger in disco:
        if ranger > 0:
            res1.append(ranger)
        else:
            dump1.append(ranger)
    for ranger in rifle:
        if ranger > 0:
            res2.append(ranger)
        else:
            dump2.append(ranger)
    if len(dump1) or len(dump2) > 0:
        print('Condition 1 and/or condition 2 are < 0')
    else:
        print('condition 1 or 2 are true')

(jumper((x-y),(x**2-y**2)))



    



