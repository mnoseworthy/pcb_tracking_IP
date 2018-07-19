"""
Test file for pseduo-coding before starting the project


Operations:
 -Convert color image to HSV color space
 -Filter noise with median Blur
 -Create mask based on HSV of filtered image for green
 -Morph mask 
 -Apply mask to original video image 
 -Canny Edge detects edges of resulting image
"""

import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()

    # Convert to HSV colorspace
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Median Blur to reduce noise (salt & peppa)
    median = cv.medianBlur(hsv,3)

    # define range of color in HSV
    lower_color = np.array([40,100,100])
    upper_color = np.array([80,255,255])

    # Threshold the HSV image
    mask = cv.inRange(median, lower_color, upper_color)

    # Morphological Operations - get struct element, choose morph
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(11,11))
    #opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    #dilation = cv.dilate(mask,kernel,iterations = 1)
    closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    #gradient = cv.morphologyEx(mask, cv.MORPH_GRADIENT, kernel)
    
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= closing)

    # Canny Edge-Detection
    edges = cv.Canny(closing,100,150)

    cv.imshow('HSV Blur',median)
    cv.imshow('original',frame)
    cv.imshow('mask',mask)
    cv.imshow('morph',closing)
    cv.imshow('result',res)
    cv.imshow('edges',edges)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()


"""
Playing with Hough Lines

Operations:
 -
"""

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