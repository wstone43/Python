word = input("write x word, any word... NO INTEGERS!: ")

wletter = []
uletter = []
lletter = []

def uplet():

  for letters in word:
    wletter.append(letters)

  print(wletter)

  for letters in wletter:
    if letters == letters.upper():
      uletter.append(letters)
    else:
      lletter.append(letters)

  print(uletter)
  print(lletter)

  

uplet()


for word in range(2):
  word = input("another word... ANY WORD!!!!!: ")

uplet()