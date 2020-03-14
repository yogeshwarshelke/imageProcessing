import numpy as np
import cv2

cap=cv2.VideoCapture("MM.mp4")  #(0)
fourcc=cv2.VideoWriter_fourcc(*"mp4v")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
#fourcc = cv2.cv.CV_FOURCC(*"XVID")
out=cv2.VideoWriter("mmoutput.mp4",fourcc,25.0,(frame_width,frame_height))  #(640,480)

while(cap.isOpened()):
    ret,frame=cap.read()
    #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Frame",gray)

    #if cv2.waitKey(20) == ord("q"):
     #   break

    if ret==True:
        #frame1=cv2.flip(frame,0)
        out.write(frame)
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1)==ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()