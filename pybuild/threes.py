import math
import numpy
import pandas

tiger = []
hun = []

ranger = range(100,100001)

for i in ranger:
    tiger.append(str(i))

for tony in tiger:
    if ((len(tony) == 3) and (tony[0:1] == '3')):
        hun.append(str(tony))
    if ((len(tony) == 3) and (tony[1:2] == '3')):
        hun.append(str(tony))
    if ((len(tony) == 3) and (tony[2:3] == '3')):
        hun.append(str(tony))
    if ((len(tony) == 4) and (tony[0:1] == '3')):
        hun.append(str(tony))
    if ((len(tony) == 4) and (tony[1:2] == '3')):
        hun.append(str(tony))
    if ((len(tony) == 4) and (tony[2:3] == '3')):
        hun.append(str(tony))
    if ((len(tony) == 4) and (tony[3:4] == '3')):
        hun.append(str(tony))

x = len(hun)
print(x)

