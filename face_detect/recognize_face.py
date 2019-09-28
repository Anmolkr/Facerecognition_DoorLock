import cv2
import numpy as np
import paho.mqtt.client as mqtt
import time
import datetime

recognizer = cv2.face_LBPHFaceRecognizer.create()
recognizer.read('trainer/trainer.yml')
cascadePath = ("cascades/haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier(cascadePath);
def on_connect(client,userdata,flags,rc):
    print("Connected with result code "+str(rc))



client = mqtt.Client()
client.will_set("die","die")
client.on_connect = on_connect
client.connect("13.235.22.232",1883)


cam = cv2.VideoCapture(0)
#font = cv2.InitFont(cv2.FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
#font = cv2.FONT_HERSHEY_SIMPLEX
#fontscale = 1
#fontcolor = (255, 255, 255)
#cv2.putText(im, str(Id), (x,y+h), fontface, fontscale, fontcolor)q
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        print(conf)
        print(Id)
        if(conf<65):
            if(Id==1):
                Id1="Anmol"
                client.loop_start()
                print("publishing")
                client.publish("/cmnd/dunit4/POWER1","OFF")

                time.sleep(4)
                client.disconnect()
                client.loop_stop()
        else:
            Id1="Unknown"
        cv2.putText(im,str(Id1), (x,y+h),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 255, 0), 2)
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()