from cv2 import *
import math
import numpy as np
import random
import sys
import Queue

nearPixel = 5
edgeSize = 3

def fillColor(inputImage, colorNum):
	ans = np.zeros((height, width, 3), np.uint8)
	colorSet = [[0,0,0]]
	for i in range(colorNum):
		colorSet.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
	for x in range(height):
		for y in range(width):
			if inputImage[x, y] != 0:
				ans[x, y] = colorSet[inputImage[x, y]]
	return ans

def bfs(ans, label, x, y):
	q = Queue.Queue(maxsize = 0)
	q.put((x, y))
	ans[x, y] = label
	while not q.empty():
		tmp = q.get()
		x, y = tmp
		shadowArea[label].append((x, y))
		flag = False
		for i in range(x - nearPixel, x + nearPixel):
			for j in range(y - nearPixel, y + nearPixel):
				if i < 0 or i >= height or j < 0 or j >= width:
					continue
				if musk[i ,j] == 0:
					if abs(i - x) < edgeSize and abs(j - y) < edgeSize:
						flag = True
					nearArea[label].append((i, j))
		if flag:
			edgeArea[label].append((x, y))
		if x + 1 < height and ans[x + 1, y] == 0 and musk[x + 1, y] != 0:
			q.put((x + 1, y))
			ans[x + 1, y] = label
		if x - 1 >= 0 and ans[x - 1, y] == 0 and musk[x - 1, y] != 0:
			q.put((x - 1, y))
			ans[x - 1, y] = label
		if y + 1 < width and ans[x, y + 1] == 0 and musk[x, y + 1] != 0:
			q.put((x, y + 1))
			ans[x, y + 1] = label
		if y - 1 >= 0 and ans[x, y - 1] == 0 and musk[x, y - 1] != 0:
			q.put((x, y - 1))
			ans[x, y - 1] = label
	return ans


def floodFill(inputImage):
	ans = np.zeros((height,width), np.uint8)
	label = 0
	for x in range(height):
		for y in range(width):
			if musk[x,y] != 0 and ans[x,y] == 0:
				label += 1
				nearArea.append([])
				shadowArea.append([])
				edgeArea.append([])
				print 'Detect No. %d' % label
				ans = bfs(ans, label, x, y)
	return ans, label

def deshadow():
	ans = img.copy()
	for i in range(1, colorNum + 1):
		shadowGrayList = [gray[pos[0], pos[1]] for pos in shadowArea[i]]
		illuminateGrayList = [gray[pos[0], pos[1]] for pos in nearArea[i]]
		sigmaL = np.std(illuminateGrayList)
		sigmaS = np.std(shadowGrayList)
		sigma = sigmaL / sigmaS
		muS = [np.mean([img[pos[0], pos[1]][tmp] for pos in shadowArea[i]]) for tmp in range(0, 3)]
		muL = [np.mean([img[pos[0], pos[1]][tmp] for pos in nearArea[i]]) for tmp in range(0, 3)]
		alpha = muL - np.dot(sigma, muS)
		# for pos in shadowArea[i]:
		# 	ans[pos] = np.dot(sigma, ans[pos]) + alpha
		for pos in shadowArea[i]:
			for r in range(3):
				tmp = ans[pos][r] * 1.0 * sigma + alpha[r]
				if tmp < 0:
					tmp = 0
				if tmp > 255:
					tmp = 255
				if math.isnan(tmp):
					tmp = 255 
				ans[pos][r] = tmp

	return ans

def deEdge():
	for i in range(1, colorNum + 1):
		for pos in edgeArea[i]:
			ans[pos] = [0,0,0]
	for i in range(1, colorNum + 1):
		for pos in edgeArea[i]:
			ans[pos[0] - 6: pos[0] + 7,pos[1] - 6: pos[1] + 7] = ans[pos[0] - 6: pos[0] + 7,pos[1] - 21: pos[1] - 8]
	return ans

def shallow():
	for x in range(height):
		for y in range(width):
			if musk[x, y] != 0:
				for r in range(3):
					tmp = ans[x,y][r] + 10
					if tmp > 255:
						tmp = 255
					ans[x,y][r] = tmp
	return ans

namedWindow('out', WINDOW_AUTOSIZE)
imgPATH = '/Users/caojingchen/Desktop/ShadowRemoval/SBU/'
imgNAME = 'lssd128'
img = imread(imgPATH + 'original/' + imgNAME + '.jpg', 1)
musk = imread(imgPATH + 'groundtruth/' + imgNAME + '.png', 0)
gray = imread(imgPATH + 'original/' + imgNAME + '.jpg', 0)
# img = imread('/Users/caojingchen/Desktop/girl.png', 1)
# musk = imread('/Users/caojingchen/Desktop/shadow.png', 0)
# gray = imread('/Users/caojingchen/Desktop/girl.png', 0)
imshow('out', musk)
waitKey(0)
height, width = musk.shape
nearArea = [[]]
shadowArea = [[]]
edgeArea = [[]]
blob, colorNum = floodFill(musk)
blob = fillColor(blob, colorNum)
imshow('out', blob)
waitKey(0)
imshow('out', img)
waitKey(0)
ans = deshadow()
ans = shallow()
imshow('out', ans)
waitKey(0)
ans = deEdge()
imshow('out', ans)
waitKey(0)
namedWindow('original', WINDOW_AUTOSIZE)
moveWindow('original', width + 100, 0)
imshow('original', img)
waitKey(0)
imwrite(imgPATH + 'output/' +imgNAME + '.png', ans)