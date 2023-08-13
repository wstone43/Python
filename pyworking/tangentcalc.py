import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-100, 100, 1)

a = float(input("select an integer for 'a': "))
b = float(input("select an integer for 'b': "))
c = float(input("select an integer for 'c': "))
e = int(input('Select an exponent: '))
d = int(input('Select tangent point: '))

x_vert = -(b/(2*a))
y = a*x**e + b*x + c
yy = a*x_vert**e + b*x_vert + c
x_vert_round = round(x_vert, 3)
y_vert = yy
y_vert_round = round(y_vert,3)
print("Your discriminant equals: ",x_vert_round)
print(f'your vertex is:',x_vert_round,',',y_vert_round)

prime = (((a*d)*(x**(e-1)))+b)
slope = (((a*e)*(d**(e-1)))+b)
tangent =(28*x)-35

tan1=slope*x

for x1 in x:
    if x1==(x1[d==x1]):
        tan2=x1
        print('x1:',tan2)
        

for y1 in x:
    if y1==(y1[d==y1]):
        y1=a*d**e+b*d+c
        tangent1=(tan1-((slope*d)-y1))
        print('y1:',y1)
        print('slope:',slope)
        

print('m*x:',(slope*d))
print('f(x):',y)
print('tangent:',tangent1)

plt.ylim(-500,500,50)
plt.grid()
plt.plot(y)
plt.plot(tangent1)
plt.show()

# **


