import cv2
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
sampleNum=0
Id = 1
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("dataset/User."+str(Id)+'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
        
        cv2.putText(img,str(sampleNum), (x, y),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 255, 0), 2)
        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>50:
        break
cam.release()
cv2.destroyAllWindows()
