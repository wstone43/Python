

cont = "yes"
cart = []

print("Welcome! please browse our list of delicious, nutritious pies! You also get softdrinks of choice cause of Chrissy!")

pies =['pecan', 'apple', 'bean', 'banoffee', 'black bun', 'blueberry', ' buko', 'burek', 'tamale', 'steak']

print(*enumerate(pies), sep='\n')


while cont == "yes":
    order = input("Enter the number of the pie you would like to order... and hurry the fuck up about it! ")
    cart.append(pies[int(order)])
    print("we gonna get you that " + pies[int(order)] +" pie plus soft drink of choice")
    cont = input("you want another one? ")

else:
    total = len(cart)
    cos = 6.99
    prod = total*cos
    dism1 = "Fine, that will be "
    dism2 = " dollars. cash or... cash?"
    
    print("finally... okay, your oder is: ")
    print(f'{dism1}{prod}{dism2}')
