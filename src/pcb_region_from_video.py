"""
Test file for pseduo-coding before starting the project


Operations:
 - Continue attempting to match a PCB until one is found in the frame
 - Once a match is found, continue tracking until it disappears
 - Repeat
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt


def hist_equalize(img):
    """
        Equalizes a color image, returning the result in color
    """
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def find_overlay_region(img, display = False):
    """
        Locates the PCB in the given image frame.
        @param img (cv2.image) - Pointer to an openCV2 image object, returned from imread().
        @param display (bool) - If set to true, the image will be displayed as the code runs, for debug usage.
        @returns [(cv2.contour), [(int, int)] - Returns a tuple containing 1) the contour object returned by opencv & 2)
            the x,y co-ordinates of the centre point of the contour
    """
    # Make a copy of image if we're going to display
    if display:
        original = img.copy()
    #img[:,:,0] = 0
    #img[:, :, 2] = 0
    # Calc image dimensions
    height, width, channels = img.shape
    area = height*width

    ## (1) Convert to gray, and threshold green channel, remove others
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Equalize histogram
    equalized = hist_equalize(img)
    hsv = cv2.cvtColor(equalized, cv2.COLOR_BGR2HSV)
    # Hue (0-180) saturation brightness
    #lower = np.array([35,100,50])
    #upper = np.array([75, 190,175])

    # Blur
    #blur = cv2.GaussianBlur(hsv, (5,5), 0)
    blur = cv2.medianBlur(hsv,5)
    #Filter w/ green threshold
    lower = np.array([45,100,100])
    upper = np.array([82, 180,180])
    green_mask = cv2.inRange(blur, lower, upper)
    green_thresh = cv2.bitwise_and(img, img, mask=green_mask )


    #morphed = threshed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(green_thresh, cv2.MORPH_CLOSE, kernel)
    # 
    edged_frame = cv2.Canny(green_thresh,100,110)

    # Gaussian blur then Otsu's thresholding
    
    #ret, threshed = cv2.threshold(blur, 50, 200, cv2.THRESH_BINARY+cv2.THRESH_OTSU)



    ## (2) Morph-op to remove noise


    ## (3) Find the max-area contour, which should be the PCB
    _, cnts, im = cv2.findContours(edged_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea)
    index = len(cnts) - 1
    cnt = False
    print("Found {} contours".format(len(cnts)))
    # Find rectangular contours
    """
    rect_cnts = []
    while index >= 0:
        perimeter = cv2.arcLength(cnts[index], True)
        approx = cv2.approxPolyDP(cnts[index], 0.01 * perimeter, True)
        if len(approx) == 4:
            print("Found rectangular contour")
            rect_cnts.append(cnts[index])
        index -= 1

    rect_cnts = sorted(cnts, key=cv2.contourArea)
    index = len(rect_cnts) - 1
    

    while index >= 0:
        #print("Image area = {}, contour area = {}".format(area*0.90, cv2.contourArea( cnts[index] )))
        if cv2.contourArea( rect_cnts[index] ) < (area*0.90):
            cnt = rect_cnts[index]
            break
    """
    x = 0
    y = 0
    if(len(cnts)):
        cnt = cnts[-1]

        #if isinstance(cnt, (int, float)) and cnt == False:
        #    print("Error finding largest contour, setting to default")
        #    cnt = rect_cnts[-1]

        # Calculate center of contour
        moment = cv2.moments(cnt)
        if(moment["m00"] != 0):
            x = int( moment["m10"] / moment['m00'] )
            y = int( moment['m01'] / moment['m00'] )

        cv2.drawContours(img, [cnt], -1, (0, 0, 255), 1)

    # Draw the contour onto the original image
    if display:
        
        #cv2.drawContours(img, rect_cnts, -1, (255, 0, 0), 3)
        output = np.hstack((original, img ))
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 800,800)
        cv2.imshow("Final output", output)
        cv2.imshow("Thresholded Image", green_thresh)
        cv2.imshow("Canny on Threshold", edged_frame)
        cv2.imshow("Morphed edge detection frame", morphed)
        #if cv2.waitKey(0) & 0xff == 27:
        #    cv2.destroyAllWindows()
    
    # Return the contour & centre coordinates
    return [cnt, [x,y] ]


def videoCapStart():
    return cv2.VideoCapture(0)

def videoCapStop(cap):
    cap.release()

def mainThread(cap):
    greenLow = 100
    greenHigh = 155

    frameCount = 0
    while(1):
        ret, frame = cap.read()
        find_overlay_region(frame, greenLow, greenHigh, True)
        frameCount = frameCount + 1
        if(frameCount % 25 == 0):
            greenLow = greenLow + 1
            greenHigh = greenHigh + 1
            frameCount = 0
        k= cv2.waitKey(5)
        if k==27:
            break
 


if __name__ == "__main__":
    # Load image
    #path = "assets/test0.jpg"
    #image = cv2.imread(path)
    

    # Print image size
    #height, width = image.shape[:2]
    #print("Input image is {} by {} ".format(height, width))

    # Get biggest contour and mask
    cap = videoCapStart()
    mainThread(cap)
    videoCapStop(cap)

