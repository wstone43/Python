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
        print("Get outta here!")
        break

test='test'