
"""
Python port of the function described below, original C++ source in comments at bottom
of file.
"""

"""

Name:       ShowManyImages

Purpose:

This is a function illustrating how to display more than one
image in a single window using Intel OpenCV

Parameters:

string title: Title of the window to be displayed
int    nArgs: Number of images to be displayed
Mat    img1: First Mat, which contains the first image
...
Mat    imgN: First Mat, which contains the Nth image

Language:   C++

The method used is to set the ROIs of a Single Big image and then resizing
and copying the input images on to the Single Big Image.

This function does not stretch the image...
It resizes the image without modifying the width/height ratio..

This function can be called like this:

ShowManyImages("Images", 5, img2, img2, img3, img4, img5);

This function can display upto 12 images in a single window.
It does not check whether the arguments are of type Mat or not.
The maximum window size is 700 by 660 pixels.
Does not display anything if the number of arguments is less than
one or greater than 12.

Idea was from [[BettySanchi]] of OpenCV Yahoo! Groups.

If you have trouble compiling and/or executing
this code, I would like to hear about it.

You could try posting on the OpenCV Yahoo! Groups
[url]http://groups.yahoo.com/group/OpenCV/messages/ [/url]


Parameswaran,
Chennai, India.

cegparamesh[at]gmail[dot]com

"""
import numpy as np
import cv2
import traceback

def ShowManyImages(title, images):
    # Check input params
    numImages = len(images)
    if numImages <= 0:
        print("Num images too small")
        return False
    elif numImages > 14:
        print("Too many images, can only handle 12 images at a time")
    
    # Grayscale flag
    gs = False
    
    # Determine size of each image in output
    # Scale map has Key=numImages, value=[Height, width, scale]
    scaleMap = {
        1 : {"h":1, "w":1, "scale":300},
        2 : {"h":1, "w":2, "scale":300},
        3 : {"h":2, "w":2, "scale":300},
        4 : {"h":2, "w":2, "scale":300},
        5 : {"h":2, "w":3, "scale":200},
        6 : {"h":2, "w":3, "scale":200},
        7 : {"h":2, "w":4, "scale":200},
        8 : {"h":2, "w":4, "scale":200},
        9 : {"h":3, "w":4, "scale":150},
        10 : {"h":3, "w":4, "scale":150},
        11 : {"h":3, "w":4, "scale":150},
        12 :{"h":3, "w":4, "scale":150}
    }

    # Create canvas image
    dims = scaleMap[numImages]
    canvas_image = np.zeros((dims['h']*dims['scale']+20*dims['w'], dims['w']*dims['scale']+20*dims['h'], 3), np.uint8)

    # Iterate over input images
    imgIndex = 1
    m = 20
    n = 20

    for img in images:

        # make copy of input
        image = img.copy()    

        # get heightxwidth of image
        if  len(image.shape) == 3 :
            x, y, channels = image.shape
            if x <= 0 or y <= 0:
                imgIndex = imgIndex + 1
                continue
        else:
            gs = True
            print("Input was grayscale")
            x, y = image.shape
            if x <= 0 or y <= 0:
                imgIndex = imgIndex + 1
                continue
        
            if gs:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Find which dimension is larger
            if  x > y :
                maxDim = x
            else:
                maxDim = y

            # Calculate scale
            scale = float(  float(maxDim) / dims['scale'] )

            # Find alignment values
            if imgIndex % dims['w'] == 0 and n != 20:
                print("Moving to next column: imgIndex {}, dims[w] {}".format(imgIndex, dims['w']) )
                n = 20
                m = m + 20 + dims['scale']

            # Calculate region to draw this image on
            height = int(y/scale)
            width = int(x/scale)
            
            # Resize image
            tmp = cv2.resize(image, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC)
            if not gs:
                _width, _height, channels = tmp.shape
            else:
                _width, _height = tmp.shape

            m_w = m+_width
            n_h = n+_height

            # Draw on canvas
            canvas_image[ m:m_w  ,  n:n_h , :] = tmp

            
        except Exception, err:
            #cv2.imshow("Failed to scale this image", image)
            traceback.print_exc()
            pass

        # Update iterators
        imgIndex = imgIndex + 1
        n = n + (20 + dims["scale"])
    

    # Create new window and display image
    cv2.imshow(title, canvas_image)


"""
void ShowManyImages(string title, int nArgs, ...) {
int size;
int i;
int m, n;
int x, y;

// w - Maximum number of images in a row
// h - Maximum number of images in a column
int w, h;

// scale - How much we have to resize the image
float scale;
int max;

// If the number of arguments is lesser than 0 or greater than 12
// return without displaying
if(nArgs <= 0) {
    printf("Number of arguments too small....\n");
    return;
}
else if(nArgs > 14) {
    printf("Number of arguments too large, can only handle maximally 12 images at a time ...\n");
    return;
}
// Determine the size of the image,
// and the number of rows/cols
// from number of arguments
else if (nArgs == 1) {
    w = h = 1;
    size = 300;
}
else if (nArgs == 2) {
    w = 2; h = 1;
    size = 300;
}
else if (nArgs == 3 || nArgs == 4) {
    w = 2; h = 2;
    size = 300;
}
else if (nArgs == 5 || nArgs == 6) {
    w = 3; h = 2;
    size = 200;
}
else if (nArgs == 7 || nArgs == 8) {
    w = 4; h = 2;
    size = 200;
}
else {
    w = 4; h = 3;
    size = 150;
}

// Create a new 3 channel image
Mat DispImage = Mat::zeros(Size(100 + size*w, 60 + size*h), CV_8UC3);

// Used to get the arguments passed
va_list args;
va_start(args, nArgs);

// Loop for nArgs number of arguments
for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {
    // Get the Pointer to the IplImage
    Mat img = va_arg(args, Mat);

    // Check whether it is NULL or not
    // If it is NULL, release the image, and return
    if(img.empty()) {
        printf("Invalid arguments");
        return;
    }

    // Find the width and height of the image
    x = img.cols;
    y = img.rows;

    // Find whether height or width is greater in order to resize the image
    max = (x > y)? x: y;

    // Find the scaling factor to resize the image
    scale = (float) ( (float) max / size );

    // Used to Align the images
    if( i % w == 0 && m!= 20) {
        m = 20;
        n+= 20 + size;
    }

    // Set the image ROI to display the current image
    // Resize the input image and copy the it to the Single Big Image
    Rect ROI(m, n, (int)( x/scale ), (int)( y/scale ));
    Mat temp; resize(img,temp, Size(ROI.width, ROI.height));
    temp.copyTo(DispImage(ROI));
}

// Create a new window, and show the Single Big Image
namedWindow( title, 1 );
imshow( title, DispImage);
waitKey();

// End the number of arguments
va_end(args);
}
"""