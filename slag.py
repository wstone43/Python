

# # import numpy as np

# # limit = int(input('select a number: '))

# # a = np.arange(-limit, limit+.5, .5)
# # b = np.arange(-limit, limit+.5, .5)

# # aprime = []
# # bprime = []
# # axb = []
# # adivb = []

# # newa = np.delete(a, np.where(a==0))
# # newb = np.delete(b, np.where(b==0))
# print('a',newa)
# print('b',newb)

# print('first',newa*newb)

# # for aa in a:
# #     if abs(aa) > .01:
# #         aprime.append(aa)
# print('aprime',aprime)

# # for bb in newb:
# #     if abs(bb) >= 1.0001:
# #         bprime.append(bb)
# print('bprime',bprime)

# print('second',aprime*bprime)

# # for shot in newa:
# #     gun = newb * shot
#     print('gun',*gun)
# #     axb.append(gun)
# print('axb', *axb)

# # for tiger in newb:
# #     woods = newa / tiger
#     print('tiger',*woods)
# #     adivb.append(tiger)
# print('adivb',tiger)

# # set1 = np.array(axb)
# print('marray',*axb)

# # set2 = np.array(adivb)
# print('darray',*adivb)

# print(np.all(set1>set2))

# import numpy as np

# # limit = int(input('select a number: '))

# # scope = limit+1

# # a = np.arange(-limit, limit+.5, .5)
# # b = np.arange(-limit, limit+.5, .5)

# print(a)
# print(b)



# # for i in range(len(a)):
# #     while i <= scope:
# #         jump = a[i] * b[i]
# print(jump)

# # for i in range(limit):
# #     while i <= scope:
# #         jump = a[i] * b[i]
# print(jump)

# ////////////////////////////////////////////////////////////////////////////////////////////////


import numpy as np

aprime = []
bprime = []
result = []

limit = int(input('select a number: '))
scope = range(-limit,limit)

a = np.arange(-limit, limit+.5, .5)
b = np.arange(-limit, limit+.5, .5)

newa = np.delete(a, np.where(a==0))
newb = np.delete(b, np.where(b==0))
# print('a',newa)
# print('b',newb)

for axb in newa:
    if abs(axb) > .01:
        aprime.append(axb)
# print('a',aprime)

for adivb in newb:
    if abs(adivb) >= 1.0001:
        bprime.append(adivb)
# print('b',bprime)

aa = np.multiply.outer(aprime,bprime)
bb = np.divide.outer(aprime,bprime)

aa.tolist()
bb.tolist()

# print('aa',aa)
# print('bb',bb)

for i in scope:
    if (aa[i][i]) > (bb[i][i]):
        result.append('ACCEPT')
        # print((aa[i][i]),(bb[i][i]))
        # print('ACCEPT')
    else:
        result.append('REJECT')
        # print((aa[i][i]),(bb[i][i]))
        # print('REJECT')

if all([results == 'ACCEPT' for results in result]):
        print('ACCEPT')

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# from mpl_toolkits.mplot3d import Axes3D

# ## Matplotlib Sample Code using 2D arrays via meshgrid
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X ** 2 + Y ** 2)
# Z = np.sin(R)
# fig = plt.figure()
# ax = Axes3D(fig)
# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# ax.set_zlim(-1.01, 1.01)

# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
 
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.title('Original Code')
# plt.show()


# # ////////////////////////////////////////////////////////////////////////////////////

# import pandas as pd

# movie_file  = 'c:/users/wston/documents/movie_scores.csv'

# movie_file_pd = pd.read_csv(movie_file)
# print(movie_file_pd)

# imdb_table = movie_file_pd[["FILM", "IMDB", "IMDB_norm",
#                             "IMDB_norm_round", "IMDB_user_vote_count"]]
# print(imdb_table)

# good_movies = movie_file_pd.loc[movie_file_pd["IMDB"] > 7, [
#     "FILM", "IMDB", "IMDB_user_vote_count"]]
# print(good_movies)

# movie_file_pd["new_metric"] = sum(movie_file_pd["IMDB"]) / len(movie_file_pd["IMDB"])

# new = movie_file_pd[["IMDB","new_metric"]]
# print(new)

# print(len(movie_file_pd))
# stat = movie_file_pd.describe()
# print(stat["IMDB"])



# //////////////////////////////////////////////////////////////////////////////////////////

# import pandas as pd
# from pandas.core.algorithms import rank



# csv_path = "c:/users/wston/documents/soccer2018data.csv"

# soccer_2018_df = pd.read_csv(csv_path, low_memory=True)
# print(soccer_2018_df)

# rankllim = soccer_2018_df.loc[soccer_2018_df["Age"] <= 28, ["Age","Overall"]]
# print("under 28 Average: ", rankllim.mean())
# print("under 28 Median: ", rankllim.median())
# print("under 28 Mode: ", rankllim.mode())

# rankulim = soccer_2018_df.loc[soccer_2018_df["Age"] > 28, ["Age","Overall"]]
# print("over 28 Average: ", rankulim.mean())
# print("over 28 Median: ", rankulim.median())
# print("over 28 Mode: ", rankulim.mode())

# new = rankllim.describe()
# mew = rankulim.describe()
# print(new)
# print(mew)

# film = {
#     "title": "Interstellar",
#     "revenues": {
#         "United States": {"Georgia": 360, "Cali": 400},
#         "China": 250,
#         "United Kingdom": 73
#     }
# }
# print(f'{film["title"]} made {film["revenues"]["United States"]["Cali"]}'" in the US.")


# ///////////////////////////////////////////////////////////////////////////////////
# from os import read
# import pandas as pd 
# import matplotlib.pyplot as plt
# import numpy as np
# import csv

# file = ('c:/users/wston/documents/soccer2018data.csv')

# n = 17981

# data = pd.read_csv(file)

# rev1 = data.drop(data[data['Age'] < 15].index)

# x = rev1.loc[:,['Age']]
# y = rev1.loc[:,['Potential']]

# print(rev1)

    
# colors = np.random.rand(n)
# plt.scatter(x,y, c = colors)
# plt.show()
        
# /////////////////////////////////////////////////////////////////////////////////

# import pandas as pd 
# import matplotlib.pyplot as plt 
# import numpy as np
# import csv

# file1 = 'c:/users/wston/documents/unemployment_2010-2011.csv'
# file2 = 'c:/users/wston/documents/unemployment_2012-2014.csv'

# new1 = pd.read_csv(file1)
# new2 = pd.read_csv(file2)

# new = pd.merge(new1, new2, on='Country Code')
# del new['Country Name_y']

# indexnew = new.rename(columns={'Country Name_x': 'Country'})
# set = indexnew.set_index('Country')

# set['C_Avg'] = set.iloc[:, 1:5].mean(axis=1)
# print(set)

# code = set.loc[:,'Country Code']
# ten = set.loc[:, '2010']
# ele = set.loc[:, '2011']
# twe = set.loc[:, '2012']
# thi = set.loc[:, '2013']
# fou = set.loc[:, '2014']
# des =set.loc[:, 'C_Avg']

# desmean = des.mean()
# desmed = des.median()
# desmode = des.mode()

# desc10 = ten.describe()
# desc11 = ele.describe()
# desc12 = twe.describe()
# desc13 = thi.describe()
# desc14 = fou.describe()
# des_C =  des.describe()

# print(desc10)
# print(desc11)
# print(desc12)
# print(desc13)
# print(desc14)
# print(des_C )

# print('Country Mean: ', desmean)
# print('Country Median: ', desmed)
# print('Country Mode: ', desmode)

# df = pd.DataFrame(ten, ele)
# print(df)

# x_axis = code
# y_axis = des

# plt.hist( y_axis, 219, density=True,alpha=1) 
# plt.show()
# plt.xticks(rotation=90)
# plt.plot(x_axis, y_axis, color='r', alpha=1) 
# plt.show()

# df.to_csv('c:/users/wston/documents/newtest.csv', index=False, header=False)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////

# def myfunc(n):
#   return lambda a : a * n

# mydoubler = myfunc(2)

# print(mydoubler(11))

# /////////////////////////////////////////////////////////////////////////////////////

# # Program to check if a number is prime or not

# num = float(input('select an integer: '))


# # To take input from the user
# #num = float(input("Enter a number: "))

# # define a flag variable
# flag = False

# # prime numbers are greater than 1
# if num > 1:
#     # check for factors
#     for i in range(2, num):
#         if (num % i) == 0:
#             # if factor is found, set flag to True
#             flag = True
#             # break out of loop
#             break

# # check if flag is True
# if flag:
#     print(num, "is not a prime number")
# else:
#     print(num, "is a prime number")

# x = 10 % 4
# print(x)

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# rain_df = pd.read_csv('c:/users/wston/documents/avg_rain_state.csv')
# rename = rain_df.rename(columns = {'Milli­metres' : 'Millimeters (MM)'})
# rename1 = rename.set_index('State')
# print(rename1)

# x_axis = np.arange(len(rename1))
# print(x_axis)
# tick_locations = [value + 0.4 for value in x_axis]

# plt.figure(figsize=(20,3))
# plt.bar(x_axis, rename1["Inches"], rename1["Rank"], color='g', alpha=0.5, align="edge")
# plt.xticks(tick_locations, rain_df["State"], rotation="vertical")
# plt.title("Average Rain per State")
# plt.xlabel("State")
# plt.ylabel("Average Amount of Rainfall in Inches")
# plt.show()

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# from mpl_toolkits.mplot3d import Axes3D

# x = np.arange(0, 100, 1)
# y = np.arange(0, 100, 1)
# z = np.arange(0, 100, 1)

# x, y, z = np.meshgrid(x, y, z)

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



# ''' We will make the board using dictionary 
#     in which keys will be the location(i.e : top-left,mid-right,etc.)
#     and initialliy it's values will be empty space and then after every move 
#     we will change the value according to player's choice of move. '''

# theBoard = {'7': ' ' , '8': ' ' , '9': ' ' ,
#             '4': ' ' , '5': ' ' , '6': ' ' ,
#             '1': ' ' , '2': ' ' , '3': ' ' }

# board_keys = []

# for key in theBoard:
#     board_keys.append(key)

# ''' We will have to print the updated board after every move in the game and 
#     thus we will make a function in which we'll define the printBoard function
#     so that we can easily print the board everytime by calling this function. '''

# def printBoard(board):
#     print(board['7'] + '|' + board['8'] + '|' + board['9'])
#     print('-+-+-')
#     print(board['4'] + '|' + board['5'] + '|' + board['6'])
#     print('-+-+-')
#     print(board['1'] + '|' + board['2'] + '|' + board['3'])

# # Now we'll write the main function which has all the gameplay functionality.
# def game():

#     turn = 'X'
#     count = 0


#     for i in range(10):
#         printBoard(theBoard)
#         print("It's your turn," + turn + ".Move to which place?")

#         move = input()        

#         if theBoard[move] == ' ':
#             theBoard[move] = turn
#             count += 1
#         else:
#             print("That place is already filled.\nMove to which place?")
#             continue

#         # Now we will check if player X or O has won,for every move after 5 moves. 
#         if count >= 5:
#             if theBoard['7'] == theBoard['8'] == theBoard['9'] != ' ': # across the top
#                 printBoard(theBoard)
#                 print("\nGame Over.\n")                
#                 print(" **** " +turn + " won. ****")                
#                 break
#             elif theBoard['4'] == theBoard['5'] == theBoard['6'] != ' ': # across the middle
#                 printBoard(theBoard)
#                 print("\nGame Over.\n")                
#                 print(" **** " +turn + " won. ****")
#                 break
#             elif theBoard['1'] == theBoard['2'] == theBoard['3'] != ' ': # across the bottom
#                 printBoard(theBoard)
#                 print("\nGame Over.\n")                
#                 print(" **** " +turn + " won. ****")
#                 break
#             elif theBoard['1'] == theBoard['4'] == theBoard['7'] != ' ': # down the left side
#                 printBoard(theBoard)
#                 print("\nGame Over.\n")                
#                 print(" **** " +turn + " won. ****")
#                 break
#             elif theBoard['2'] == theBoard['5'] == theBoard['8'] != ' ': # down the middle
#                 printBoard(theBoard)
#                 print("\nGame Over.\n")                
#                 print(" **** " +turn + " won. ****")
#                 break
#             elif theBoard['3'] == theBoard['6'] == theBoard['9'] != ' ': # down the right side
#                 printBoard(theBoard)
#                 print("\nGame Over.\n")                
#                 print(" **** " +turn + " won. ****")
#                 break 
#             elif theBoard['7'] == theBoard['5'] == theBoard['3'] != ' ': # diagonal
#                 printBoard(theBoard)
#                 print("\nGame Over.\n")                
#                 print(" **** " +turn + " won. ****")
#                 break
#             elif theBoard['1'] == theBoard['5'] == theBoard['9'] != ' ': # diagonal
#                 printBoard(theBoard)
#                 print("\nGame Over.\n")                
#                 print(" **** " +turn + " won. ****")
#                 break 

#         # If neither X nor O wins and the board is full, we'll declare the result as 'tie'.
#         if count == 9:
#             print("\nGame Over.\n")                
#             print("It's a Tie!!")

#         # Now we have to change the player after every move.
#         if turn =='X':
#             turn = 'O'
#         else:
#             turn = 'X'        
    
#     # Now we will ask if player wants to restart the game or not.
#     restart = input("Do want to play Again?(y/n)")
#     if restart == "y" or restart == "Y":  
#         for key in board_keys:
#             theBoard[key] = " "

#         game()

# if __name__ == "__main__":
#     game()


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# x = float(input("select a number you would like to square: "))

# def square(num: float) -> float:
#     sq = num * num
#     return sq
# print(square(x))



# x = int(input("select an integer to add to your square: "))

# sum = x + y

# print(sum)

# # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# run = 'y'

# a = []
# b = []
# c = []



# def append(word, list):
#     x = input('select a word: ')
#     y = input('select a list: ')
#     if list == 'a':
#         a.append(word)
#     if list == 'b':
#         b.append(word)
#     if list == 'c':
#         c.append(word)
# append(x, y)

# print(a)