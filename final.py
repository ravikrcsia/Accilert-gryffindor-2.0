import json
import geopy.distance

import serial
import time
from decimal import *
from subprocess import call

# import RPi.GPIO as GPIO      
import os, time
from random import randrange

with open(r"hub_coords.json", mode='r') as read_file:
    data = json.load(read_file)

data=list(data)
for x in data:
    print(x)

flag=True

def trigger(data, la1, lo1):

    for c,x in enumerate(data,0):
        coords_1 = (la1, lo1)
        coords_2 = (x[1][0], x[1][1])
        d=geopy.distance.geodesic(coords_1, coords_2)
        p="{dis}".format(dis=d)
        q=p.replace(" km", "")
        x.append(float(q))

    low_dis=data[0][3]
    index=0

    for c,x in enumerate(data,0):
        if x[3]<low_dis:
            low_dis=x[3]
            index=c

    #print("\nNearest Location: {} \nContact : {} \nDistance : {} km".format(data[index][0], data[index][2], low_dis))
    return low_dis, index


#GPS operation here:
def find(str, ch):
    for i, ltr in enumerate(str):
        if ltr == ch:
            yield i

def gps():
    # Enable Serial Communication
    port = serial.Serial("/dev/ttyUSB0", baudrate=115200, timeout=1)
    # Transmitting AT Commands to the Modem
    # '\r\n' indicates the Enter key
    
    port.write('AT'+'\r\n')            
    rcv = port.read(100)
    print(rcv)
    time.sleep(.1)
    
    port.write('AT+CGNSPWR=1'+'\r\n')             # to power the GPS
    rcv = port.read(100)
    print(rcv)
    time.sleep(.1)
    
    port.write('AT+CGNSIPR=115200'+'\r\n') # Set the baud rate of GPS
    rcv = port.read(100)
    print(rcv)
    time.sleep(.1)
    
    port.write('AT+CGNSTST=1'+'\r\n')    # Send data received to UART
    rcv = port.read(100)
    print(rcv)
    time.sleep(.1)
    
    port.write('AT+CGNSINF'+'\r\n')       # Print the GPS information
    rcv = port.read(200)
    print(rcv)
    time.sleep(.1)
    ck=1
    while ck==1:
        fd = port.read(200)        # Read the GPS data from UART
        #print fd
        time.sleep(.5)
        if '$GNRMC' in fd:        # To Extract Lattitude and 
            ps=fd.find('$GNRMC')        # Longitude
            dif=len(fd)-ps
            if dif > 50:
                data1=fd[ps:(ps+50)]
                print(data1)
                ds=data1.find('A')        # Check GPS is valid
                if ds > 0 and ds < 20:
                    p=list(find(data, ","))
                    lat=data[(p[2]+1):p[3]]
                    lon=data[(p[4]+1):p[5]]
    
    # GPS data calculation

                    s1=lat[2:len(lat)]
                    s1=Decimal(s1)
                    s1=s1/60
                    s11=int(lat[0:2])
                    s1 = s11+s1
    
                    s2=lon[3:len(lon)]
                    s2=Decimal(s2)
                    s2=s2/60
                    s22=int(lon[0:3])
                    s2 = s22+s2
    
                    return s1,s2

#SMS/GSM operation here
def sms(mob_no, la,lo,dis):

    # GPIO.setmode(GPIO.BOARD)    
    
    # Enable Serial Communication
    port = serial.Serial("/dev/ttyUSB0", baudrate=115200, timeout=1)
    
    # Transmitting AT Commands to the Modem
    # '\r\n' indicates the Enter key
    port.write('AT'+'\r\n')
    rcv = port.read(10)
    print(rcv)
    time.sleep(.1)
    
    port.write('ATE0'+'\r\n')      # Disable the Echo
    rcv = port.read(10)
    print(rcv)
    time.sleep(.1)
    
    port.write('AT+CMGF=1'+'\r\n')  # Select Message format as Text mode
    rcv = port.read(10)
    print(rcv)
    time.sleep(.1)
    
    port.write('AT+CNMI=2,1,0,0,0'+'\r\n')   # New SMS Message Indications
    rcv = port.read(10)
    print(rcv)
    time.sleep(.1)
    
    # Sending a message to a particular Number
    port.write('AT+CMGS={}'.format(str(mob_no))+'\r\n')
    rcv = port.read(10)
    print(rcv)
    time.sleep(.1)
    
    port.write('''
    Latitude :{}
    Longitude :{}
    Google Map link :https://www.google.co.in/maps/place/{},{}
    Distance :{} km
    '''.format(la,lo,la,lo,dis)+'\r\n')  # Message
    rcv = port.read(10)
    print(rcv)
    
    port.write("\x1A") # Enable to send SMS
    for i in range(10):
        rcv = port.read(10)
        print(rcv)

while(True and flag):
    key=input("Enter 1 Test : ")
    print(type(key))
    if(int(key)==1):
        key=True
        flag=False #to stop this loop for next execution !
    else:
        key=False

    if(key==True):
        lat, lon=gps()
        ld, ind=trigger(data, lat, lon)
        print("\nNearest Location: {} \nContact : {} \nDistance : {} km".format(data[ind][0], data[ind][2], ld))
        # sms(data[ind][2], lat, lon, ld)
        print("Help is on it WAY !!!")

