import numpy as np
import cv2

cv2.namedWindow('show', cv2.WINDOW_AUTOSIZE)
img = cv2.imread('/Users/caojingchen/Desktop/1.jpg',1)
im = img.astype(np.float32)+0.001 #to avoid division by 0
c1c2c3 = np.arctan(im/np.dstack((cv2.max(im[...,1], im[...,2]), cv2.max(im[...,0], im[...,2]), cv2.max(im[...,0], im[...,1]))))
cv2.imshow('show',c1c2c3)
cv2.waitKey(0)
height, width, _ = c1c2c3.shape
shadow = np.zeros((height,width), np.uint8)
print c1c2c3[10,10]
for x in range(height):
	for y in range(width):
		if (c1c2c3[x,y][2] * 1.0 / (c1c2c3[x,y][0] + c1c2c3[x,y][1] + c1c2c3[x,y][2]) > 0.6) and c1c2c3[x,y][2] > 0:
			shadow[x,y] = 255

cv2.imshow('show',shadow)
cv2.waitKey(0)