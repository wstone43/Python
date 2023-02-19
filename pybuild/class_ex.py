planes = ('737-900ER', 
'747-400', '757-200',
'777-200LR', 'A320neo', 
'A321neo', 'A330-200', 
'A330-300', 'A350-900')

for fleet in planes:
    print(f'[{str(planes.index(fleet))}] {fleet}')

pilot = input('Select a jet to fly ')

i = [pilot]
u = []

for index in i:
    u.append(planes[int(index)])

class aircraft:
        
    def __init__(jet, engine, man):
        jet.engine = engine
        jet.man = man
        

    def welcome(jet):
        print('welcome aboard...' + 'A ' +jet.engine + ' built by ' + jet.man)

        print(f'')
                                                                                            

class velocity:

    def __init__(speed):
        speed.mach = input('Enter MACH speed: ')

    def vel(speed):
        print('you will be traveling at MACH ' + speed.mach)
        
j1 = aircraft('turbofan', 'Boeing')
j2 = aircraft('piece of shit', 'airbus')
v1 = velocity()

if i == 1:
    j1.welcome
else:
    j2.welcome


print('You will be piloting a ')
for ii in u:
    print(ii)



v1.vel()




def face(m):
    m = int(input('select a speed (as MACH) '))
    print('you will be traveling at MACH' + m)

