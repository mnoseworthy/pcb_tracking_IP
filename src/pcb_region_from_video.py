"""
Test file for pseduo-coding before starting the project


Operations:
 - Continue attempting to match a PCB until one is found in the frame
 - Once a match is found, continue tracking until it disappears
 - Repeat
"""


import cv2
import numpy as np
import sys
import traceback
import argparse
from time import sleep
from threading import Event
from multi_display import ShowManyImages

# Adjust default behavior
FROM_CAM = False
VID_FILE = 0
STEP_THROUGH_FRAMES = False
ONE_FRAME_ONLY = False
DISPLAY_ALL = False

class pcb_region_detection():
    def __init__(self, video_stream=None):
        ###################################################################
        #   Section: Define attributes
        ###################################################################
        # Capture object from opencv
        self.cap = None
        self.frame = None
        self.total_video_frames = None
        self.frame_count = 1

        # Processing buffer
        self.buffer = {
            "Input"     : None,
            "equalized" : None,
            "bgr"       : None,
            "grayscale" : None,
            "hsv"       : None,
            "blurred"   : None,
            "thresholded" : None,
            "morphed_close" : None,
            "morphed_open" : None,
            "edged" : None,
            "contours" : None,
            "hough" : None,
            "Output" : None
        }

        # Processing error flag
        self.failure = False

        # Flag to determine how to display results
        # False = Display everything in separate windows
        # True = Display all 
        self.display = None

        # List of function pointers which the input frame is to be passed through
        self.function_pipe = [
            self.hist_equalize,
            self.bgr_to_hsv,
            self.blur_before_thresh,
            self.hsv_green_thresholding,
            
            self.morphology_operation,
            self.hsv_to_gray,
            self.canny_edge_detection,
            #self.morphology_operation,
            #self.hough_line_trans,
            self.contour_filter
        ]

        ###################################################################
        #   Section: Define init control flow
        ###################################################################
        # Obtain/start video stream
        if isinstance(video_stream, bool):
            self.videoCapStart()
        else:
            self.cap = video_stream
    
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
        #self.buffer["equalized"] = img
        return self.buffer["equalized"]
    
    def bgr_to_hsv(self, img):
        """
            Converts a bgr array to hsv format
        """
        self.buffer["hsv"] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return self.buffer["hsv"]
    
    def hsv_to_gray(self, img):
        """
            Converts HSV image to Grayscale
        """
        bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        self.buffer["grayscale"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return self.buffer["grayscale"]

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
        #lower = np.array([45,25,0])
        #upper = np.array([82, 140,180])
        lower = np.array([45,70,70])
        upper = np.array([82, 255, 255])
        
        #lower = np.array([0,100,100])
        #upper = np.array([180, 180,180])
        green_mask = cv2.inRange(img, lower, upper)
        self.buffer["thresholded"] = cv2.bitwise_and(img, img, mask=green_mask )
        return self.buffer["thresholded"]

    def morphology_operation(self, img):
        """
            Uses morphology to attempt to reduce salt and pepper noise
        """

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
        self.buffer["morphed_close"] = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        self.buffer["morphed_open"] = cv2.morphologyEx(self.buffer["morphed_close"], cv2.MORPH_OPEN, kernel)
        return self.buffer["morphed_open"]

    def hough_line_trans(self, img):
        """
            Applys hough line transform
        """
        im = img.copy()
        lines = cv2.HoughLines(im,1,np.pi/180,200)
        #print(type(lines))
        if isinstance(lines, np.ndarray):
            for line in lines:
                for rho,theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))

                    cv2.line(im,(x1,y1),(x2,y2),(255,255,255),5)
        self.buffer["hough"] = im
        return img
    
    def canny_edge_detection(self, img):
        """
            Performes canny edge detection on the image and returns the result
        """
        sigma = 0.33
        v = np.median(img)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(max(0, (1.0 + sigma) * v))
        self.buffer["edged"] = cv2.Canny(img,lower,upper)
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

        if(len(cnts)>0):
            return cnts[-1]
        else:
            self.failure = True
            return False


    def find_overlay_region(self, frame=None):
        """
            Locates the PCB in the given image frame.
            @param img (cv2.image) - Pointer to an openCV2 image object, returned from imread().
            @param display (bool) - If set to true, the image will be displayed as the code runs, for debug usage.
            @returns [(cv2.contour), [(int, int)] - Returns a tuple containing 1) the contour object returned by opencv & 2)
                the x,y co-ordinates of the centre point of the contour
        """
        # Init failure flag
        self.failure = False
        # Check if we need to pull a new frame
        if not isinstance(self.frame, np.ndarray):
            print("pcb region detection had to get own frame, consider passing a frame if using for tracking")
            #print(type(self.frame))
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
                if isinstance(result, str) and result == "Unset":
                    result = funct(self.frame)
                else:
                    result = funct(result)

                if not isinstance(result, np.ndarray):
                    #print("Type returned was {}".format(type(result)))
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

            #self.buffer["Output"] = cv2.drawContours(self.frame, [result], -1, (0, 0, 255), 1)
            self.buffer["Output"] = self.frame
            # Value error very possible here, catch later when finishing this code
            x, y, w, h = cv2.boundingRect(result)
            bbox = (x, y, w, h)
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(self.buffer["Output"], p1, p2, (0,255,0), 2, 1)

            # Return the contour & centre coordinates
            ret = [result, [x,y] ]

        else:
            self.buffer["Output"] = self.frame

        # If true, output all intermediate images as well as Input/Output
        if self.display == True: 
            
            # Add text to images
            self.addText(self.buffer["Input"], "Input")
            self.addText(self.buffer["Output"], "Output")
            self.addText(self.buffer["equalized"], "Equalized")
            self.addText(self.buffer["morphed_open"], "Morphed Open")
            self.addText(self.buffer["morphed_close"], "Morphed Close")
            self.addText(self.buffer["thresholded"], "Thresholded")
            self.addText(self.buffer["blurred"], "Median Blur")
            self.addText(self.buffer["edged"], "edged")
            self.addText(self.buffer["hsv"], "HSV")
            #self.addText(self.buffer["hough"], "hough transform")
            img_list = [
                self.buffer["Input"],  
                self.buffer["equalized"],
                self.buffer["hsv"],
                self.buffer["blurred"],
                self.buffer["thresholded"],
                self.buffer["morphed_close"], 
                self.buffer["morphed_open"], 
                self.buffer["edged"],
                self.buffer["Output"]            
            ]
            
            ShowManyImages("images", img_list)
        # If false output just Input/Output frames
        elif self.display == False:
            cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Output", self.buffer["Output"])

   
        
        
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

    def videoCapStart(self, source=0):
        #print(source)
        self.cap = cv2.VideoCapture(source)
        # Update # of frames in video
        self.total_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self.cap

    def videoCapStop(self):
        self.cap.release()
    
    def getFrame(self):
        ret, self.frame = self.cap.read()
        while not ret:
            print("Could not get video frame...\n Either the video path doesn't exist, or no webcam is available.")
            ret, self.frame = self.cap.read()
            sleep(1)
        #print("Ret from frame pull: {}".format(ret))
        self.frame_count = self.frame_count + 1
        return self.frame

    def mainThread(self, source=0):

        if not FROM_CAM:
            self.videoCapStart(VID_FILE)
        else:
            self.videoCapStart(0)
        if ONE_FRAME_ONLY:
            frame = self.getFrame()
            self.find_overlay_region(frame)
            Event().wait()
        else:
            while(1):
                try:
                    frame = self.getFrame()
                    self.find_overlay_region(frame)
                    
                    
                    if STEP_THROUGH_FRAMES:
                        #If the last frame is reached, reset the capture and the frame_counter                   
                        print( "Frame {}/{}".format( self.frame_count, self.total_video_frames) )                       
                        # Wait forever for a keypress
                        k = cv2.waitKey(0)
                        # Exits when spacebar pressed, quits when esc pressed
                        while k != 32:
                            if k == 27:
                                sys.exit()
                            k = cv2.waitKey(0)
                    else:
                        # Just check if esc pressed
                        k= cv2.waitKey(5)
                        if k==27:
                            break
                    
                    #Handle looping video
                    if self.frame_count == self.total_video_frames:
                        self.frame_count = 1 #Or whatever as long as it is the same as next line
                        if not FROM_CAM:
                            self.videoCapStart(VID_FILE)
                        else:
                            self.videoCapStart(0)

                except Exception, err:
                    traceback.print_exc()
                    self.videoCapStop()
                    break
            return False
    



if __name__ == "__main__":
    # Add argparser
    parser = argparse.ArgumentParser(description='PCB Region Detection')
    parser.add_argument('--from-cam', dest='FROM_CAM', action='store_true', help='Set to use camera feed')
    parser.add_argument('--step-through-frame', dest='STEP_THROUGH_FRAME', action='store_true', help='If set, press space to move through frames, esc to exit.')
    parser.add_argument('--one-frame-only', dest='ONE_FRAME_ONLY', action='store_true', help='Runs a single frame through, and halts until esc pressed.')
    parser.add_argument('--display-all', dest='DISPLAY_ALL', action='store_true', help='If set will display intermediate processing frames.')
    parser.add_argument('video_path', help='path to video file if not using camera')
    # Parse args with some nice ugly if's
    ARGS = parser.parse_args()
    print(ARGS)
    if ARGS.FROM_CAM:
        FROM_CAM = True
    if ARGS.STEP_THROUGH_FRAME:
        STEP_THROUGH_FRAMES = True
    if ARGS.ONE_FRAME_ONLY:
        ONE_FRAME_ONLY = True
    if ARGS.DISPLAY_ALL:
        DISPLAY_ALL = True
    VID_FILE = ARGS.video_path
    
    # Create object and start main thread
    try:
        pcb_det = pcb_region_detection()
        pcb_det.display = DISPLAY_ALL
        pcb_det.mainThread()
    finally:
        pcb_det.videoCapStop()
