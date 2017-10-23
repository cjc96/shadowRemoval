from cv2 import *
import numpy as np

shadow = imread('/Users/caojingchen/Desktop/out.jpg')
namedWindow('original')
imshow('original', shadow)
shadow = morphologyEx(shadow, MORPH_CLOSE, np.ones((15,15),np.uint8))
shadow = morphologyEx(shadow, MORPH_OPEN, np.ones((11,11),np.uint8))

# shadow = GaussianBlur(shadow, (5, 5), 0);

namedWindow('out')
moveWindow('out', shadow.shape[1] + 100, 0)
imshow('out', shadow)
waitKey(0)
imwrite('/Users/caojingchen/Desktop/out2.jpg', shadow)