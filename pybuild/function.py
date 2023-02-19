def square(num):
    sq = num * num
    return sq
square(5)



def squares(num):
    square = num * num 
    return square
    
squares(5)

f = lambda x:x**2  
print(f(5))


def even(a):
    new_list = []
    for i in a:
        if i%2 == 0:
            new_list.append(i)
    return(new_list)
    
a = [1,2,3,4,5]
even(a)

x = float(input("select a number you would like to square: "))

def square(num: float) -> float:
    sq = num * num
    print(sq)
    print(type(x))
square(x)


sum = square(x)+z
print(sum)

