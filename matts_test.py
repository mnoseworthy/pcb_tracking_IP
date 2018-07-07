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


# Edge detection to handle object matching
def find_largest_contour(framePath):
    img = cv2.imread(framePath,0)
    edges = cv2.Canny(img,100,200)

    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()


if __name__ == "__main__":
    path = "assets/test0.jpg"
    find_largest_contour(path)

