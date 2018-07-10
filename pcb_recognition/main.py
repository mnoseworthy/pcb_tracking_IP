"""
Testing PyTorch supervised learning, using the worst possible dataset, randomly
downloaded pictures that match a few given keywords.
"""


from torch.utils.data.dataset import Dataset
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy
import traceback

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

            # 
            self.imgBuffer = _image_buffer(self.num_images)

        def __getitem__(self, index):
            label = self.images[index]
            img = self.imgBuffer.__get()
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
                        self.images.append(file)
            # initialize file count
            self.num_images = len(self.images)
        
        def parseImages(self):
            """
                @brief Load images into numpy arrays and preform initial
                filtering in parallel as results are stored in the self.imgBuffer
                attribute
            """




if __name__ == "__main__":
    try:

        #ds = MyCustomDataset()
        #ds.getImages()
        with BufferPoolResource(20, "batched") as bufPool:

            bufObjs = []
            for i in range(20):
                bufObjs.append( bufPool.make() )

            
            for i, obj in enumerate(bufObjs):
                bufPool.writeout( i, obj)

    except Exception, err:
        traceback.print_exc()


