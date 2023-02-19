
import os
import csv

new = []


file = os.path.join('c:/users/wston/documents/iadalist.csv')
newpath = os.path.join('c:/users/wston/documents/new.csv')


with open(file, newline='') as read:
    csv_read = csv.reader(read, delimiter =',')
    print(csv_read)

    # csv_header = next(csv_read)
    # print(f"CSV Header: {csv_header}")

    for row in csv_read:
        print(row)

new.append(csv_read)

for row in new:
    print(new)

with open(newpath, 'w', newline='') as newfile:
    writer = csv.writer(newfile, delimiter=',')
    writer.writerow(new)