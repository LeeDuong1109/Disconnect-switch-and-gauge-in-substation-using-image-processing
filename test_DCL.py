import os
import cv2
import numpy as np
from skimage.feature import canny
from skimage import morphology
import function as ff
import DCL_processing as DCL

path = '/NCKH/nckh/DCL'
list = os.listdir(path)
i = 0
for img_path in list:
    i+=1
    print("number: {0}".format(i))
    DCL.horizontal_test(img_path)
