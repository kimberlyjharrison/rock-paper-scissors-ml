import cv2
import numpy as np


cap = cv2.VideoCapture(0)


while(True):

    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    kernel = np.ones((3,3),np.uint8)

    
    #define region of interest
    roi=frame[100:500, 100:500]
    cv2.rectangle(frame,(100,100),(500,500),(0,255,0),3)    
    
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,"Show Rock, Paper or Scissors in Box",(0,40), font, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame,"PRESS 'c' TO CAPTURE",(0,90), font, 1, (0,0,255), 1, cv2.LINE_AA)



    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', (800, 1200))
    cv2.imshow('mask', mask)
    cv2.imshow('frame', frame)
    cv2.moveWindow("mask", 20,20);
    cv2.moveWindow("frame", 500,0);
    
    if cv2.waitKey(1) & 0xFF == ord('c'):
        out = cv2.imwrite('capture.png', mask)
        img = cv2.imread('capture.png')
        img = cv2.resize(img, (50,50))
        out2 = cv2.imwrite('cap.png',img)
        print('image captured')
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()