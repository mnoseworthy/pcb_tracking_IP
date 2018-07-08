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
import traceback

from multi_display import ShowManyImages

class pcb_region_detection():
    def __init__(self):
        ###################################################################
        #   Section: Define attributes
        ###################################################################
        # Capture object from opencv
        self.cap = None

        # Most recently captured frame
        self.frame = None

        # Processing buffer
        self.buffer = {
            "Input" : None,
            "equalized" : None,
            "blurred" : None,
            "thresholded" : None,
            "morphed" : None,
            "edged" : None,
            "contours" : None,
            "Output" : None
        }

        # Processing error flag
        self.failure = False

        # Flag to determine weather or not to display results
        # False = Just display output frame
        # True = Display all 
        # None = no display
        self.display = None

        # List of function pointers which the input frame is to be passed through
        self.function_pipe = [
            self.hist_equalize,
            self.bgr_to_hsv,
            self.blur_before_thresh,
            self.hsv_green_thresholding,
            self.morphology_operation,
            self.canny_edge_detection,
            #self.hsv_to_gray,
            self.contour_filter
        ]

        ###################################################################
        #   Section: Define init control flow
        ###################################################################
        self.videoCapStart()
    
    def clearBuffers(self):
        """
            Emptys buffer data
        """
        for key, value in self.buffer.iteritems():
            self.buffer[key] = None

    def hist_equalize(self, img):
        """
            Equalizes a color image, returning the result in color
        """
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        self.buffer["equalized"] = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return self.buffer["equalized"]
    
    def bgr_to_hsv(self, img):
        """
            Converts a bgr array to hsv format
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    def hsv_to_gray(self, img):
        """
            Converts HSV image to Grayscale
        """
        bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

    def blur_before_thresh(self, img):
        """
            Blurs an image, to be used before inputing the return through
            a thresholding algorithm
        """
        self.buffer["blurred"] = cv2.medianBlur(img, 5)
        return self.buffer["blurred"]

    def hsv_green_thresholding(self, img):
        """
            Uses a HSV format image, and thresholds based on hue to remove all colors
            but the range of greens we expect a PCB to be
        """
        lower = np.array([45,100,100])
        upper = np.array([82, 180,180])
        green_mask = cv2.inRange(img, lower, upper)
        self.buffer["thresholded"] = cv2.bitwise_and(img, img, mask=green_mask )
        return self.buffer["thresholded"]

    def morphology_operation(self, img):
        """
            Uses morphology to attempt to reduce salt and pepper noise
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        self.buffer["morphed"] = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return self.buffer["morphed"]
    
    def canny_edge_detection(self, img):
        """
            Performes canny edge detection on the image and returns the result
        """
        self.buffer["edged"] = cv2.Canny(img,100,110)
        return self.buffer["edged"]

    def contour_filter(self, img):
        """
            After all image processing operations have completed, we're left with an 
            array of contours that we must filter to output our best guess at which
            contour contains the PCB.
        """
        #height, width, channels = img.shape
        #area = height*width
        _, cnts, im = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        if(len(cnts)):
            return cnts[-1]
        else:
            self.failure = True
            return False


    def find_overlay_region(self):
        """
            Locates the PCB in the given image frame.
            @param img (cv2.image) - Pointer to an openCV2 image object, returned from imread().
            @param display (bool) - If set to true, the image will be displayed as the code runs, for debug usage.
            @returns [(cv2.contour), [(int, int)] - Returns a tuple containing 1) the contour object returned by opencv & 2)
                the x,y co-ordinates of the centre point of the contour
        """
        # Check if we need to pull a new frame
        if not isinstance(self.frame, list):
            self.getFrame()

        # Clear buffers
        self.clearBuffers()

        # Setup return buffer [Contour Matrix, [Center X, Center Y]] 
        ret = [False, [False, False]]

        # Make a copy of image if we're going to display
        if self.display != None:
            self.buffer["Input"] = self.frame.copy()

        # Push frame through the function pipe, result should be a single contour
        try:
            result = "Unset"
            for funct in self.function_pipe:
                # First function requires input frame
                if result == "Unset":
                    result = funct(self.frame)
                else:
                    result = funct(result)

                if not isinstance(result, np.ndarray):
                    print("Type returned was {}".format(type(result)))
                    self.failure = True
                    print("Failure occured during function {}".format(funct) )
                    break
        except Exception, err:
            traceback.print_exc()
            self.failure = True
            pass
        
        if not self.failure:
            # Calculate center of contour
            moment = cv2.moments(result)
            if(moment["m00"] != 0):
                x = int( moment["m10"] / moment['m00'] )
                y = int( moment['m01'] / moment['m00'] )

            self.buffer["Output"] = cv2.drawContours(self.frame, [result], -1, (0, 0, 255), 1)

            # If true, output all intermediate images as well as Input/Output
            if self.display == True: 
                # Add text to images
                self.addText(self.buffer["Input"], "Input")
                self.addText(self.buffer["Output"], "Output")
                self.addText(self.buffer["equalized"], "equalized")
                self.addText(self.buffer["morphed"], "morphed")
                self.addText(self.buffer["thresholded"], "thresholded")
                self.addText(self.buffer["blurred"], "blurred")
                self.addText(self.buffer["edged"], "edged")
                img_list = [
                    self.buffer["Input"],
                    self.buffer["Output"],
                    self.buffer["equalized"],
                    self.buffer["morphed"],
                    self.buffer["thresholded"],
                    self.buffer["blurred"],
                    self.buffer["edged"]
                ]
                
                ShowManyImages("images", img_list)
            # If false output just Input/Output frames
            elif self.display == False:
                cv2.imshow("Input", self.buffer["Input"])
                cv2.imshow("Output", self.buffer["Output"])
   
            # Return the contour & centre coordinates
            ret = [result, [x,y] ]

        
        # Clear processed frame
        self.frame = None

        # Return values
        return ret

    def addText(self, img, text):
        """
            Adds text to upper-left corner of image
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        top_left = (50,50)
        fontScale = 2
        fontColor = (255,255,255)
        lineType = 2
        cv2.putText(img, text, top_left, font, fontScale, fontColor, lineType)

    def videoCapStart(self):
        self.cap = cv2.VideoCapture(0)
        return self.cap

    def videoCapStop(self):
        self.cap.release()
    
    def getFrame(self):
        ret, self.frame = self.cap.read()
        return self.frame

    def mainThread(self):
        self.display = True
        while(1):
            try:
                self.getFrame()
                self.find_overlay_region()
                cv2.waitKey(0)
            except Exception, err:
                traceback.print_exc()
                self.videoCapStop()
                break
        return False
 


if __name__ == "__main__":
    # Create object and start main thread
    pcb_det = pcb_region_detection()
    pcb_det.mainThread()

