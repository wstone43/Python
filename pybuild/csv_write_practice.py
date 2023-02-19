import os
import csv 

file = os.path.join('..', 'python', 'web_starter2.csv')

header = ['TITLE', 'PRICE', 'SUB COUNT', 'REVIEW NUMBER', 'COURSE LENGTH']
title = []
price = []
subcnt = []
norev = []
length = []

with open(file, newline="") as csvfile:
    read = csv.reader(csvfile, delimiter=',')

    for col in read:
        title.append(col[0])
        price.append(col[1])
        subcnt.append(col[2])
        norev.append(col[3])
        length.append(col[4])
    print(title)
    print(price)
    print(subcnt)
    print(norev)
    print(length)

tupp = (title, price, subcnt, norev, length)

i = zip(tupp)

outlet = os.path.join('..', 'python', 'newfile.csv')

with open(outlet, 'w', newline="") as csvfile:
    write = csv.writer(csvfile)
    write.writerow(header)
    write.writerows(i)