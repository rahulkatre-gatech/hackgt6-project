import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # It converts the BGR color space of image to HSV color space 
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
      
    # Threshold of a color
    lower = np.array([0, 0, 30]) 
    upper = np.array([80, 80, 255]) 
  
    # preparing the mask to overlay 
    mask = cv2.inRange(frame, lower, upper) 
      
    # The black region in the mask has the value of 0, so when multiplied with original image removes all non-matching regions 
    result = cv2.bitwise_and(frame, frame, mask = mask) 
  
    # Display the resulting frames
    cv2.imshow('frame', frame) 
    cv2.imshow('mask', mask) 
    cv2.imshow('result', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
