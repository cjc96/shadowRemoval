#coding=utf8
from cv2 import *
import math
import numpy as np
import matplotlib.pyplot as plt

# 转换成HSV，注意，opencv中H值域为[0,179]，要换成[0,255]
def normalHSV(inputImage):
	outputImage = cvtColor(inputImage, COLOR_BGR2HSV)
	height, width, _ = outputImage.shape
	for x in range(outputImage.shape[0]):
		for y in range(outputImage.shape[1]):
			outputImage[x,y][0] = math.trunc(outputImage[x,y][0] * 255.0 / 179.0)
	return outputImage

# 利用大津法(Ostu)计算阈值 
def ostu(inputImage):
	outputImage = inputImage
	height, width = outputImage.shape
	minGray,maxGray = 255, 0
	for x in range(height):
		for y in range(width):
			if outputImage[x,y] > maxGray:
				maxGray = outputImage[x,y]
			if outputImage[x,y] < minGray:
				minGray = outputImage[x,y]
	maxThredshold = 0.0
	ans = 0.0
	tick = 0
	for tmpThreshold in range(minGray, maxGray + 1):
		tick += 1
		iforeground = 0
		ibackground = 0
		foreground = 0.0
		background = 0.0

		for x in range(height):
			for y in range(width):
				if outputImage[x,y] >= tmpThreshold:
					iforeground += 1
					foreground += outputImage[x,y]
				else:
					ibackground += 1
					background += outputImage[x,y]
		if iforeground != 0 and ibackground != 0:
			w0 = 1.0 * iforeground / height / width
			w1 = 1.0 * ibackground / height / width
			u0 = foreground / iforeground
			u1 = background / ibackground
			tmpAns = w0 * w1 * (u0 - u1) * (u0 - u1)
			if tmpAns > maxThredshold:
				maxThredshold = tmpAns
				ans = tmpThreshold
			matplotList.append(tmpAns)
		if tick % 10 == 0:
			print 'tick : %d; tmpans = %f' % (tick,tmpAns)
	return int(ans)


# img = imread('/Users/caojingchen/Desktop/3.jpg', 1)
matplotList = []
img = imread('/Users/caojingchen/Downloads/743487.jpg', 1)
namedWindow('origin', WINDOW_AUTOSIZE)
namedWindow('hsv', WINDOW_AUTOSIZE)
namedWindow('shadow', WINDOW_AUTOSIZE)
imshow('origin', img)
hsv = normalHSV(normalHSV(img))
height, width, _ = hsv.shape
moveWindow('hsv', 0, height + 100)
moveWindow('shadow', width + 100, 0)
imshow('hsv',hsv)
waitKey(0)

hsvGray = cvtColor(hsv, COLOR_BGR2GRAY)
threshold = ostu(hsvGray)
shadow = np.zeros((height,width), np.uint8)
for x in range(height):
	for y in range(width):
		if hsvGray[x, y] < threshold:
			shadow[x,y] = 255
		else:
			shadow[x,y] = 0;

# imwrite('/Users/caojingchen/Desktop/out.jpg',shadow)
plt.plot(matplotList)
plt.ylabel('STD')
plt.xlabel('gray threshold')
plt.show()
imshow('shadow', shadow)

waitKey(0)