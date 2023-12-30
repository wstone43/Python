import os
import csv
import pandas as pd

path = os.path.join('c:/users/wston/documents/movie1.csv')
file = 'c:/users/wston/documents/movie1.csv' 

name = input('Select an actor: ')

with open(path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    # print(reader)

    # for row in reader:
    #     print(row)

frame = pd.read_csv(file)
# pd.set_option('max_columns', None)
print(frame)

# for i in frame: 
#     if name in i:
#         print('Yes')
#     else:
#         print('No')

if name in frame:
    print('Yes')
else:
    print('No')
