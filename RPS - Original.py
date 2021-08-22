import random

Comp = ("R", "p", "S")

n = 3
r = 3 

pscore = 0
cscore = 0

x = input("Select 'R', 'P', or 'S'  *CASE Sensitive* ")

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
    pscore += 1
    print("YOU WIN!")
elif x == "P" and y == "R":
    pscore += 1
    print("YOU WIN!")
elif x == "S" and y == "P":
    pscore += 1
    print("YOU WIN!")
elif x == "R" and y == "P":
    cscore += 1
    print("Sorry, the computer has won...")
elif x == "P" and y == "S":
    cscore += 1
    print("Sorry, the computer has won...")
elif x == "S" and y == "R":
    cscore += 1
    print("Sorry, the computer has won...")
elif x == y:
    print("TIE! Please play again!")
else:
    print("INVALID SELECTION. PLEASE TRY AGAIN.")

print("---SCORES---")
print("PLAYER:", pscore)
print("COMPUTER:", cscore)

z = input("play again? enter 'Y' if so. ")


while  z  == "Y":
    
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
        pscore += 1
        print("YOU WIN!")
    elif x == "P" and y == "R":
        pscore += 1
        print("YOU WIN!")
    elif x == "S" and y == "P":
        pscore += 1
        print("YOU WIN!")
    elif x == "R" and y == "P":
        cscore += 1
        print("Sorry, the computer has won...")
    elif x == "P" and y == "S":
        cscore += 1
        print("Sorry, the computer has won...")
    elif x == "S" and y == "R":
        cscore += 1 
        print("Sorry, the computer has won...")
    elif x == y:
        print("TIE! Please play again!")
    else:
        print("INVALID SELECTION. PLEASE TRY AGAIN.")

    print("---SCORES---")
    print("PLAYER:", pscore)
    print("COMPUTER:", cscore)

    z = input("play again? enter 'Y' if so. ")





