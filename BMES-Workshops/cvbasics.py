import cv2 as cv 
import numpy as npy 

img = cv.imread(r"C:\Users\Death\Downloads\ezgif.com-gif-maker-3.jpg") #Add any image in here
#vid = cv.VideoCapture(0) #setup for webcam can put in any mp4 video to work as well

haar_cascade = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
#haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml') 
#download required=download file from github (haarcascade_frontalface_default.xml)

path = cv.CascadeClassifier(haar_cascade)
#cv.imshow('FinalIMG',img)

#gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#blur=cv.GaussianBlur(img, (5,5), cv.BORDER_REFLECT) #try switching img to gray 
#edges=cv.Canny(img, .5,.8) #try using gray or blur instead of img

#color1=cv.cvtColor(img, cv.COLOR_BGR2HSV) try other color channel conversions
faces = path.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1, minSize=(20,20), flags=cv.CASCADE_SCALE_IMAGE)

#For video can add the for loop below inside for detection in video 
'''
while True:
        ret, frame = vid.read()

        if not ret:
                break

                cv.imshow('Frame', vid)

        if cv.waitKey(5) & 0xFF == 27:
                break
'''
for (x,y,w,h) in faces:
                    
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv.putText(img, 'Face', (x+6,y-6), cv.FONT_HERSHEY_COMPLEX, .5, (0,255,0), 1)

cv.imshow('Final', img)
cv.waitKey(0)

#vid.release()
#cv.destroyAllWindows()
