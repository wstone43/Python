import random
import requests

Comp = ("R","P","S")


x = input("Select 'R', 'P', or 'S'  ")

if x == "R":
    print("Player chooses 'Rock'")
elif x == "P":
    print("Player chooses 'Paper'")
elif x == "S":
    print("Player chooses 'Scisors'")

y = random.choice(Comp)

if y == "R":
    print("Computer chooses 'Rock'")
elif y == "P":
    print("Computer chooses 'Paper'")
elif y == "S":
    print("Computer chooses 'Scisors'")

if x == "R" and y == "S":
    print("YOU WIN!")
elif x == "P" and y == "R":
    print("YOU WIN!")
elif x == "S" and y == "P":
    print("YOU WIN!")
elif x == "R" and y == "P":
    print("Sorry, the computer has won...")
elif x == "P" and y == "S":
    print("Sorry, the computer has won...")
elif x == "S" and y == "R":
    print("Sorry, the computer has won...")
elif x == y:
    print("TIE! Please play again!")
else:
    print("INVALID SELECTION. PLEASE TRY AGAIN.")







