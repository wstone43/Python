import random

x = []
y = [] 

w = []
n = []
s = []

x = "y"


while x == "y":
    
    i = random.randint(0,100000001)

    if i <= 25000000:    
        w.append(1)
        print("weak")       
    elif i <= 75000000:
        n.append(1)
        print("normal")        
    elif i <= 100000000:
        s.append(1)
        print("strong")
    else:
        print("overpower")
        next
        x = input("print 'y' to run again!: ")
        
        
     

    print(i)
    print("Weak:" + str(sum(w)))
    print("Normal:" + str(len(n)))
    print("Strong:" + str(sum(s)))


