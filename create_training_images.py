## Script to create images used to train CNN model for rock, paper, scissors
## Contributors: K. Harrison, H. Orrantia

## Import dependencies
import cv2
import numpy as np
import os

## set image size
image_x, image_y = 50, 50

## initalize webcam
cap = cv2.VideoCapture(0)

## create folder to store images
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def main(g_id):

    ## create folder to store images
    create_folder("gestures/" + str(g_id))
    counter = 601

    while True:
        ## create frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        ## create mask settings
        kernel = np.ones((3,3),np.uint8)

        #define region of interest
        roi=frame[100:500, 100:500]
        cv2.rectangle(frame,(100,100),(500,500),(0,255,0),3)    
        
        ## set color properties (note: should match final rps game settings)
        ## create mask
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))   

        ## loop to captures images when user presses 'c' key
        if cv2.waitKey(1) & 0xFF == ord('c'):
            ## write image to path
            path = f'gestures/{g_id}/{counter}.png'
            cv2.imwrite(path, mask)

            ## print image number, incriment image number
            print(f'image captured #{counter}')
            counter = counter + 1

        ## show frame and mask optimazed for 13" macbook screen
        ## suggested to put terminal window in bottom left corner of screen
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', (800, 1200))
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)
        cv2.moveWindow("mask", 20,20);
        cv2.moveWindow("frame", 500,0);

## print user for gesture type to set up folder structure
g_id = input("Enter gesture: ")
main(g_id)

