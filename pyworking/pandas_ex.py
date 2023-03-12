import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

file1 = 'c:/users/wston/documents/unemployment_2010-2011.csv'
file2 = 'c:/users/wston/documents/unemployment_2012-2014.csv'

new1 = pd.read_csv(file1)
new2 = pd.read_csv(file2)

new = pd.merge(new1, new2, on='Country Code')
del new['Country Name_y']

indexnew = new.rename(columns={'Country Name_x': 'Country'})
set = indexnew.set_index('Country')

set['C_Avg'] = set.iloc[:, 1:5].mean(axis=1)
print(set)

code = set.loc[:,'Country Code']
ten = set.loc[:, '2010'].describe()
ele = set.loc[:, '2011'].describe()
twe = set.loc[:, '2012'].describe()
thi = set.loc[:, '2013'].describe()
fou = set.loc[:, '2014'].describe()
des =set.loc[:, 'C_Avg']

desmean = des.mean()
desmed = des.median()
desmode = des.mode()


print(code)
print(ten)
print(ele)
print(twe)
print(thi)
print(fou)
print(des)

print('Country Mean: ', desmean)
print('Country Median: ', desmed)
print('Country Mode: ', desmode)

x_axis = code
y_axis = des

plt.hist( y_axis, 219, density=True,alpha=1) 
plt.show()
plt.xticks(rotation=90)
plt.plot(x_axis, y_axis, color='r', alpha=1) 
plt.show()