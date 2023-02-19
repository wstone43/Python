from tkinter import *

window = Tk()

window.title("default white box")

window.mainloop()



rock = PhotoImage(file='C:/Users/wston/pictures/rock.png')
paper = PhotoImage(file='C:/Users/wston/pictures/paper.png')
scissors = PhotoImage(file='C:/Users/wston/pictures/wes.png')

image_list = [rock, paper, scissors]

pick_number = random.randint(0,2)

rando = random.randint(0, 10)

image_label = Label(window, image=image_list[pick_number])
image_label.pack(pady=2)

# ////////////////////////////////////////

button_rock.place(x=0, y=0)
slogan = tkinter.Button(frame,
                   text="Hello",
                   command=write)
slogan.place (x=0, y=0)

button_rock = tkinter.Button(frame, 
                   text="ROCK!", 
                   fg="red",
                   command=quit)


# ///////////////////////////////////////////////////////////


messagebox.showinfo( "Hello Python", "Hello World")

def (release):
    input(str( def))

# ///////////////////////////////////////////////////////////

print("Welcome Capitan!")

mylist = ("747", "737", "757", "777")

firstclass = input("how many first class passengers do you have?")
coach = input("how many coach passengers do you have?")
flightrange = input("what is the distance you are flying?")


pax = int(firstclass) + int(coach)
flightrange= int(flightrange)

if (pax < 100) and (flightrange < 3000):
    print(mylist[1])
if (pax >= 100) and (pax < 200) and (flightrange >=3000) and (flightrange < 7000):
    print(mylist[3])
if (pax >= 200) and (flightrange >= 7000 ) and ((flightrange < 9500 )):
    print(mylist[0])

else:
    print("No Applicable Equipment")

# ///////////////////////////////////////////////////////////

IMAGE SELECTION:

pick_number = random.randint(0,2)

image_label = Label(window, image=image_list[pick_number])
image_label.pack(pady=2)


# ////////////////////////////////////////////////////////////////


Comp = ("R", "R", "S")



n = 3
r = 3 


pscore = 0
cscore = 0




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

z = input("play again? enter 'y' if so. ")


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


    # ///////////////////////////////////////////////////////////////////////////

n = type(int)

while n == int:
    n = input("select a number range." + " defined as 0...n ")
     
    print("you did not select an integer- PLEASE TRY AGAIN!")


    # ///////////////////////////////////////////////////////////////////////////

x = int(input('Yo! what number you wanna go to, Alan Turing? '))

for n in range(x+1):
    print(n)

a =input('you wanna do that again? smash the "y" button if so. if not press "n" ') 

while a == "y": 

    x = int(input('Yo! what number you wanna go to, Alan Turing?'))

    for n in range(x+1):
        print(n)

    a =input('you wanna do that again? smash the "y" button if so. if not press "n" ')
    if a == "n":
        break 