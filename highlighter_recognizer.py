import math
import numpy as np
import cv2

import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    imgarr = plt.subplots(1, 3)[1]

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
    
    x_vals = []
    if cv2.findNonZero(mask) is not None:
        x_vals.append(cv2.findNonZero(mask)[0][0][0])
    vals = np.array(x_vals)

    if not math.isnan(np.median(vals)):
        print(int(np.median(vals)))

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
