"""
    Using methods in other modules, a region of interest is available. This module will use that region
    to create a tracking object that should track the object while it moves in the video stream
"""
import cv2
import numpy as np
import sys
from time import sleep

from src.pcb_region_from_video import  pcb_region_detection as reg_det

FROM_CAM = False
VID_FILE = "./assets/tracking1.MOV"



if __name__ == "__main__":

    # Set up tracker.
    # Instead of MIL, you can also use
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]

    tracker = cv2.TrackerTLD_create()


    # Read video
    if FROM_CAM:
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(VID_FILE)
    sleep(1)



    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print( 'Cannot read video file')
        sys.exit()
    
    # Init ROI detection
    ROI_det = reg_det(video)
    ROI_det.display = None # Set true to see frame output from region detection
    
    # Define an initial bounding box
    contour, (midXY) = ROI_det.find_overlay_region(frame)
    # Value error very possible here, catch later when finishing this code
    x, y, w, h = cv2.boundingRect(contour)
    bbox = (x, y, w, h)
    


    # Uncomment the line below to select a different bounding box
    #bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    cv2.namedWindow("Template matching tracker", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Template matching tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
        
        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
    
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        # Check output from frame ROI
        contour, (midXY) = ROI_det.find_overlay_region(frame)
        # Value error very possible here, catch later when finishing this code
        x, y, w, h = cv2.boundingRect(contour)
        bbox = (x, y, w, h)
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)

        # Display result
        cv2.imshow("Template matching tracker", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
    