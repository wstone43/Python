import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-500, 500, .01)

xlist=[]
xlist.append(x)

a = float(input("select an integer for 'a': "))
b = float(input("select an integer for 'b': "))
c = float(input("select an integer for 'c': "))

x_vert = -(b/(2*a))
y = a*x**2 + b*x + c
yy = a*x_vert**2 + b*x_vert + c
x_vert_round = round(x_vert, 3)
y_vert = yy
y_vert_round = round(y_vert,3)
print("Your discriminant equals: ",x_vert_round)
print(f'your vertex is:',x_vert_round,',',y_vert_round)

plt.ylim(-500,500,10)
plt.grid()
plt.plot(x,y)
plt.show()





