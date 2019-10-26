import pyautogui
import math
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

width =  cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

x_vals = []
y_vals = []

while(True):
    if len(x_vals) == 10:
        x_vals.clear()

    if len(y_vals) == 10:
        y_vals.clear()

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # It converts the BGR color space of image to HSV color space 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
      
    # Threshold of a color
    lower = np.array([20, 100, 100]) 
    upper = np.array([30, 255, 255]) 
  
    # preparing the mask to overlay 
    mask = cv2.inRange(hsv, lower, upper)
    #  mask = cv2.fastNlMeansDenoising(mask, mask)

    contours = cv2.findContours(image = mask, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(image = mask, contours = contours, contourIdx = -1, color = (0, 0, 255), thickness = 5)
    
    blob = cv2.flip(mask, 1)

    
    if cv2.findNonZero(blob) is not None:
        x_vals.append(cv2.findNonZero(blob)[0][0][0])
        y_vals.append(cv2.findNonZero(blob)[0][0][1])
    x = np.array(x_vals)
    y = np.array(y_vals)

    position = ()
    if not math.isnan(np.median(x)) and not math.isnan(np.median(y)):
        position = ((int(np.median(x)), int(np.median(y))))
        pyautogui.moveTo(3 * int(position[0]), 3 * int(position[1]), duration = 0)
    
    print(position)

    # The black region in the mask has the value of 0, so when multiplied with original image removes all non-matching regions 
    # result = cv2.bitwise_and(frame, frame, mask = mask) 

    # Display the resulting frames
    # cv2.imshow('frame', frame) 
    cv2.imshow('Mask', cv2.flip(mask, 1)) 
    # cv2.imshow('result', cv2.flip(result, 1))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
