"""
Testing PyTorch supervised learning, using the worst possible dataset, randomly
downloaded pictures that match a few given keywords.
"""

# Stdlib
import os
from os import listdir
from os.path import isfile, join
import traceback
from multiprocessing import Process, Lock
# Pip requirements
from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
# Project source
from src.ts_buf_pool import BufferPoolResource

class PCB_Dataset(Dataset):
        def __init__(self):
            """
                Initializer
            """

            # Path to image assets
            self.imgDir = "/home/matt/projects/google-images-download/downloads"

            # Updated by @getImages
            self.num_images = None
            # Updated by @getImages
            self.images = []

            # Empty, will be filled after initial processing complete
            self.imgBuffer = []

            # Updated by callback from buffer pool
            self.pp = True

            # Trigger pre-processing control flow
            self.getImages()
            self.parseImages()

        def __getitem__(self, index):
            label = self.images[index].split('/')[-1]
            img = self.imgBuffer[index]
            return (img, label)
        
        def __len__(self):
            return self.num_images

        def getImages(self):
            """
                @Brief walks the directory under imgDir and creates a 
                list of all the filepaths, placed in self.imagesPaths
            """
            # Get all dem files
            for subDir in os.walk(self.imgDir):
                files = [f for f in listdir(subDir[0]) if isfile(join(subDir[0], f))]
                for file in files:
                    if file.endswith(".jpg"):
                        if isinstance(subDir[0], str):
                            x = os.path.join(subDir[0], file)
                            self.images.append(x)
            # initialize file count
            self.num_images = len(self.images)
            #print(self.images)
        
        def parseImages(self):
            """
                @brief Load images into numpy arrays and preform initial
                filtering in parallel as results are stored in the self.imgBuffer
                attribute
            """
            # Buffer pool must be used with `with` keyword to properly handle memory,
            # so our callback must be within scope 
            # Define simple callback to copy the whole buffer pool's memory to our dataset
        
            def callback(data, status_check=False):
                if status_check:
                    return self.pp
                else:
                    print("Callback ran")
                    self.images = data
                    print(data)
                    self.pp = False

            # Init a lock for threads to use during buffer requests
            lock = Lock()

            with BufferPoolResource(self.num_images, "batched",  callback) as BufferPool:
                self.BufferPool = BufferPool
                threads = []
                # Iterate over each image path and spawn a thread for each
                i = 0
                for image in self.images:
                    t = Process( name=str(image), target=self.threadWorker ,  args=( str(image), lock, int(i)   )  )
                    t.start()
                    i = i + 1
                # Wait for buffer batch to complete
                while not callback(None, True):
                    pass
                
                

                for t in threads:
                    t.join()
        
        def threadWorker(self, imagePath, lock, index):
            """
                Defines the task each thread will run during pre-processing the data-set
            """
            if self.BufferPool:
                # Claim a buffer in the pool
                buf = self.BufferPool.make(lock, index) # Use image path as seed, as it is unique
                # Load image
                img = cv2.imread(imagePath)
                # Preform pre-processing 
                # Writeout and close thread
                self.BufferPool.writeout(img, buf)





if __name__ == "__main__":
    try:

        ds = PCB_Dataset()
        #ds.getImages()
        #with BufferPoolResource(20, "batched") as bufPool:

        #    bufObjs = []
        #    for i in range(20):
        #        bufObjs.append( bufPool.make() )

            
        #    for i, obj in enumerate(bufObjs):
        #        bufPool.writeout( i, obj)


    except Exception, err:
        traceback.print_exc()


