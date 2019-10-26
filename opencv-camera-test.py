import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for i in gray:
        if i < 100:
            i /= 4
        else:
            i *= 2

    # It converts the BGR color space of image to HSV color space 
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      
    # Threshold of a color
    lower = np.array([30, 30, 70])
    upper = np.array([162, 182, 217])
  
    # preparing the mask to overlay 
    mask = cv2.inRange(gray, 40, 160)

    contours = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(image=mask, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=5)

    lines = cv2.Canny(gray, 40, 45)
  
    # Display the resulting frames
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    cv2.imshow('canny', lines)
    cv2.imshow('result', mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
