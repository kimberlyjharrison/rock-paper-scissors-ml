import cv2
import numpy as np
import os

image_x, image_y = 50, 50

cap = cv2.VideoCapture(0)

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def main(g_id):
    total_pics = 1200
    cap = cv2.VideoCapture(0)
    x, y, w, h = 300, 50, 350, 350

    create_folder("gestures/" + str(g_id))
    counter = 601

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)

        
        #define region of interest
        roi=frame[100:500, 100:500]
        cv2.rectangle(frame,(100,100),(500,500),(0,255,0),3)    
        
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))   

        
        if cv2.waitKey(1) & 0xFF == ord('c'):
            path = f'gestures/{g_id}/{counter}.png'
            cv2.imwrite(path, mask)
            print(f'image captured #{counter}')
            counter = counter + 1

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', (800, 1200))
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)
        cv2.moveWindow("mask", 20,20);
        cv2.moveWindow("frame", 500,0);


g_id = input("Enter gesture: ")
main(g_id)

