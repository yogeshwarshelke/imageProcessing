import numpy as np
import cv2

#main code
count=0
img=cv2.imread("/home/yogesh/PycharmProjects/imageprocessing/multiple.jpg",1)
facecascad=cv2.CascadeClassifier('/home/yogesh/PycharmProjects/imageprocessing/haarcascades/haarcascade_frontalface_default.xml')
#eyecascad=cv2.CascadeClassifier('/home/yogesh/Desktop/haarcascades/haarcascade_eye.xml')
faces=facecascad.detectMultiScale(img,1.3,2)

for (x,y,w,h) in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_color=img[y:y+h,x:x+w]
    count+=1
    #eys Detaction
    """ eyes=eyecascad.detectMultiScale(roi_color)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    """
cv2.imshow("image", img)

#exit
if cv2.waitKey(0)==ord('q'):
    print(count)
    cv2.destroyAllWindows()
elif cv2.waitKey(0)==ord("s"):
    print(count)
    cv2.imwrite("/home/yogesh/Desktop/dataset/salman.jpeg",img)
    cv2.destroyAllWindows()
