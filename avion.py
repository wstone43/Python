import os 
import sys


wslist = os.path.join('C:\\Users\\wston\\Documents\\pythonwp',"wslist.txt")

with open(wslist) as f:
    roster = f.read()
    print(roster) 

pilots = ("Tony Soprano", "Paulie Walnutz", "Uncle Junior", "Silvio Dante", "Johnny Sacramoni", "Bobby Bacalieri")
rank = (" ", "Capitan", "F/O", "Reserve Capitan", "Reserve F/O")
jets = ("747", "737", "757", "777", "No Applicable Equipment Type")



Pilot_ID = int(input("Welcome, Please Enter Your Identifier: "))

ident = int(Pilot_ID) - 1000

if (ident == 11):
    print()
    print("welcome "  + pilots[1])
elif (ident == 12):
    print()
    print("welcome "  + pilots[2])
elif (ident == 13):
    print()
    print("welcome "  + pilots[3])
elif (ident == 14):
    print()
    print("welcome "  + pilots[4])
elif (ident == 15):
    print()
    print("welcome "  + pilots[5])
elif (ident == 16):
    print()
    print("welcome "  + pilots[6])
else:
     sys.exit("INVALID I.D. - Please try again...")

print()
print()
print()

Flight = int(input("Enter Your Flight #: "))
  
pilotrank = input("Please enter your rank: 1) Capitan 2) F/O 3) Reserve Capitan 4) Second Officer ")

rankid = int(pilotrank)

if (rankid == 1):
        print(rank[1])
        print("FLIGHT RESPONSIBILITIES: ")
        print("1) P.I.C. SIGN-OFF")
        print("2) MANIFEST CHECK")
        print("3) FUEL ORDER")
        print("4) CREW BRIEF")
        print("5) ROTATION ASSIGNMENTS")
    

elif (rankid == 2):
        print(rank[2])
        print("WALK-AROUND")
        print("FMC LOAD")
        print("CLEARANCE DELIVERY")
        print("F/A CROSSCHECK")
        print("ALL DISCRETIONARY ORDERS BY CAPITAN")

elif (rankid == 3):
        print(rank[3])
        print("DISCRETION OF CAPITAN")

elif (rankid == 4):
        print(rank[4])
        print("DISCRETION OF CAPITAN")

else: 
        sys.exit()

sign = int(input("Please enter your PILOT ID to concent to your command: "))

if sign == Pilot_ID:
                print("YOU HAVE CONCENTED TO COMMANDIND FLIGHT", Flight)

else: 
                print("INVALID CONCENT - PLEASE BEGIN AGAIN... ")

    
firstclass = input("how many first class passengers do you have? ")
coach = input("how many coach passengers do you have? ")


pax = int(firstclass) + int(coach)


if (pax < 150): 
    print(jets[1]) 
if (pax >= 150) and (pax < 300): 
    print(jets[3]) 
if (pax >= 300): 
    print(jets[0])



flightrange = input("what is the distance you are flying?")

flightrange= int(flightrange)

if (pax < 150) and (flightrange < 3000):
    print(jets[1])
if (pax < 300) and (flightrange >= 3000) and (flightrange <7000):
    print(jets[3])
if (pax <= 500) and (flightrange >= 7000) and (flightrange <=11000):
    print(jets[0])

else:
   print(jets[4])

mach = input("what is your intended MACH number?")


Time = int(flightrange) / (float(mach) * 768)

hours = int(Time)
minutes = (Time*60) % 60

print("______________________________________________________")
print(" ")
print(("your estimated flight time is ") + (("%d:%02d" % (hours, minutes))))
print(" ")
print(" ")