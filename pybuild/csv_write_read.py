import os 
import csv

csvpath = os.path.join('..', 'documents', 'test')

with open(csvpath, 'w', newline = '') as csvfile:



    csvwriter = csv.writer(csvfile, delimiter=',')


    csvwriter.writerow(['First Name', 'Last Name', 'em_id'])
    csvwriter.writerow(['Caleb', 'Frost', '1'])
    csvwriter.writerow(['Wesley', 'Stone', '2'])
    csvwriter.writerow(['John','Mills','3'])
    csvwriter.writerow(['Don','Childs','4'])
    csvwriter.writerow(['Tony','Koufax','5'])
    csvwriter.writerow(['Eric','Hughes','6'])
    csvwriter.writerow(['Sam','Pullman','7'])

with open(csvpath, newline='') as csvfile:
   
    csvreader = csv.reader(csvfile, delimiter = ',')

    print(csvreader)

    csv_header = next(csvreader)
    print(f'CSV Header: {csv_header}')

    for row in csvreader:
        print(row)


# ////////ALTERNATE READ FORMAT:////////
# with open(csvpath, 'r') as file_handler:
#     lines = file_handler.read()
#     print(lines)
#     print(type(lines))