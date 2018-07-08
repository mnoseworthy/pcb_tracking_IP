"""import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()"""

import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of color in HSV
    lower_color = np.array([50,100,50])
    upper_color = np.array([255,150,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_color, upper_color)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()

"""import numpy as np
import cv2
cap = cv2.VideoCapture(0)

while(1):
    
    _, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(frame,50,150,apertureSize = 3)

    minLineLength=100
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)

    a,b,c = lines.shape
    for i in range(a):
        cv2.line(frame, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
        #cv2.line(gray, pt1, pt2, (0, 0, 255), 3)

    cv2.imshow('frame',frame)
    #cv2.imshow('gray',gray)
    #cv2.imshow('threshed',threshed)
    cv2.imshow('edges',edges)
    
    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()"""