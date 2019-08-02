## Script to play rock, paper, scissors using Computer Vision and a CNN model
## Contributors: K. Harrison, H. Orrantia

## Import dependencies
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

## Load model that was created and trained in previous step
model = load_model('cnn_rps.h5')

## initalized OpenCV webcam
cap = cv2.VideoCapture(0)

## translate 0, 1, or 2 to paper, rock, scissors (rsepectively)
def num_to_rps(num):
    if num == 0:
        return("PAPER")
    elif num == 1:
        return("ROCK")
    elif num == 2:
        return("SCISSORS")
    else:
        return("Please Try Again")

## define function to take user input and "play" Rock, Paper, Scissors
def playRPS(user):
    ## generate random interger between 0-2 inclusive for computer choice;
    ## follow rock, paper, scissors logic
    cpu = np.random.randint(0, 3)
    if user == 0 and cpu == 1:
        return 'User Wins!', cpu
    elif user == 0 and cpu == 2:
        return 'CPU Wins!', cpu
    elif user == 1 and cpu == 2:
        return 'User Wins!', cpu
    elif user == 1 and cpu == 0:
        return 'CPU Wins!', cpu
    elif user == 2 and cpu == 0:
        return 'User Wins!', cpu
    elif user ==2 and cpu == 1:
        return 'CPU Wins!', cpu
    elif user == cpu:
        return "Draw", cpu
    else:
        return "Please Try Again", cpu

## initalized screen with empty choices and instruct user how to play
user_choice = ""
cpu_choice = ""
text_result = "Press C To Play" 

while(True):

    ## create frame
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)

    #define region of interest & draw rectangle around region
    roi=frame[100:500, 100:500]
    cv2.rectangle(frame,(100,100),(500,500),(0,255,0),3)    
    
    ## crate mask to show black/white image of region of interest
    kernel = np.ones((3,3),np.uint8)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))

    ## loop to captures images when user presses 'c' key
    if cv2.waitKey(1) & 0xFF == ord('c'):
        ## save unprocessed image
        out = cv2.imwrite('capture.png', mask)
        print('User Input Captured!')

        ## define path to image just created
        image_path = 'capture.png'

        ## load image as 50x50 pixels
        img = image.load_img(image_path, target_size=(50,50), color_mode='grayscale')

        ## convert to array & expand dims
        user_img = image.img_to_array(img)
        user_img = np.expand_dims(user_img, axis=0)

        ## create prediction using CNN model
        predict = model.predict_classes(user_img)[0]

        ## pass results to RPS function & translate results
        text_result, cpu_choice = playRPS(predict)
        user_choice = num_to_rps(predict)
        cpu_choice = num_to_rps(cpu_choice)
        print(text_result)

    ## create black rectangles on frame for readability
    cv2.rectangle(frame, (0,0), (650, 100), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (0,500), (650, 750), (0, 0, 0), cv2.FILLED)
    
    ## add text to frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,"User Chose: ",(0,600), font, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, user_choice,(200,600), font, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame,"Computer Chose: ",(0,650), font, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, cpu_choice,(300,650), font, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, text_result, (0,700), font, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame,"Show Rock, Paper or Scissors in Box",(0,40), font, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame,"PRESS 'c' TO PLAY",(0,90), font, 1, (0,0,255), 2, cv2.LINE_AA)
    
    ## show frame and mask optimazed for 13" macbook screen
    ## suggested to put terminal window in bottom left corner of screen
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', (800, 1200))
    cv2.imshow('mask', mask)
    cv2.imshow('frame', frame)
    cv2.moveWindow("mask", 20,20);
    cv2.moveWindow("frame", 500,0);
    
## release and destroy windows 
cap.release()
cv2.destroyAllWindows()