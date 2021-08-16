from os import write
import random

from tkinter import *


window = Tk()

frame = Frame(window)
frame.pack()

window.geometry('800x1200')
window.config(bg="blue")
window.title("Rock, Paper, Scissors!")

rando = print(str(random.randint(0, 10)))

a = Button(window, width = 20, height = 20, text = "random number", command = rando)
b = Button(window, width = 20, height = 5, text = "Fuck This!", command = quit)

a.place(x=400, y=100)
b.place(x=800, y=1200)
















window.mainloop()