# ///////////////////////PYTHON LOOPER//////////////////////////


from tkinter import *

root = Tk()
frame = Frame(root)
frame.pack()

bottomframe = Frame(root)
bottomframe.pack( side = BOTTOM )

redbutton = Button(frame, text="Red", fg="red")
redbutton.pack( side = LEFT)

greenbutton = Button(frame, text="Brown", fg="brown")
greenbutton.pack( side = LEFT )

bluebutton = Button(frame, text="Blue", fg="blue")
bluebutton.pack( side = LEFT )

blackbutton = Button(bottomframe, text="Black", fg="black")
blackbutton.pack( side = BOTTOM)
root.mainloop()
    
a = "y" or "Y"

while a == "y" or a == "Y":
    try:
        x = int(input('Yo! what number you wanna go to, Alan Turing? '))
    except ValueError:
        print("thats not an integer, idiot!")
        try:
            x1 = int(input("enter an integer, you oobatz! "))
        except ValueError:
                print("still not an integer, you oobatz^2")
            
        else:
            for n in range(x1+1):
                print(n)
    else:
        for n in range(x+1):
            print(n)
    

    a =input('you wanna do that again? smash the "y" button if so. if not press "n" ') 

    if a == "n" or a == "N":
        print("fuck outta here!")
        break
    

root.mainloop()