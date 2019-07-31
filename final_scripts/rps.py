import cv2
import numpy as np

from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('cnn.h5')


cap = cv2.VideoCapture(0)


def playRPS(user):
    cpu = np.random.randint(0, 3)
    print(cpu)
    if user == 0 and cpu == 2:
        return 'User Wins!', cpu
    elif user == 0 and cpu == 1:
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

def num_to_rps(num):
    if num == 0:
        return("PAPER")
    elif num == 1:
        return("ROCK")
    elif num == 2:
        return("SCISSORS")
    else:
        return("Please Try Again")

user_choice = ""
cpu_choice = ""
text_result = "Press C To Play" 

while(True):

    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    kernel = np.ones((3,3),np.uint8)

    
    #define region of interest
    roi=frame[100:500, 100:500]
    cv2.rectangle(frame,(100,100),(500,500),(0,255,0),3)    
    
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))


    if cv2.waitKey(1) & 0xFF == ord('c'):
        out = cv2.imwrite('capture.png', mask)
        print('image captured')

        image_path = 'capture.png'
        img = image.load_img(image_path, target_size=(50,50), color_mode='grayscale')
        user_img = image.img_to_array(img)
        user_img = np.expand_dims(user_img, axis=0)
        predict = model.predict_classes(user_img)[0]

        text_result, cpu_choice = playRPS(predict)
        user_choice = num_to_rps(predict)
        cpu_choice = num_to_rps(cpu_choice)

    cv2.rectangle(frame, (0,0), (650, 100), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (0,500), (650, 750), (0, 0, 0), cv2.FILLED)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,"User Chose: ",(0,600), font, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, user_choice,(200,600), font, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame,"Computer Chose: ",(0,650), font, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, cpu_choice,(300,650), font, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, text_result, (0,700), font, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame,"Show Rock, Paper or Scissors in Box",(0,40), font, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame,"PRESS 'c' TO PLAY",(0,90), font, 1, (0,0,255), 1, cv2.LINE_AA)
    
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', (800, 1200))
    cv2.imshow('mask', mask)
    cv2.imshow('frame', frame)
    cv2.moveWindow("mask", 20,20);
    cv2.moveWindow("frame", 500,0);
    
cap.release()
cv2.destroyAllWindows()