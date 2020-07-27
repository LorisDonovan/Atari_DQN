import torch
import matplotlib.pyplot as plt 
import numpy as np
import cv2 

from replayMemory import Experience


def extractTensors(experiences):
	batch = Experience(*zip(*experiences))
	t1 = torch.cat(batch.state)
	t2 = torch.cat(batch.action)
	t3 = torch.cat(batch.reward)
	t4 = torch.cat(batch.nextState)
	return (t1, t2, t3, t4)

def preprocessing(img):
	img = cv2.cvtColor(cv2.resize(img, (84, 110)), cv2.COLOR_BGR2GRAY)
	img = img[18:102, ] # crop
	# _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
	# return np.reshape(img, (1, 1, 84, 84))
	return torch.tensor(img/255).reshape(1, 1, 84, 84).to(torch.float32)

def plotSubplot(frames = None, num = 4, nrow = 1, ncol = 4, titles = None):
	for i in range(num):
		plt.subplot(nrow, ncol, i+1)
		if titles is not None:
			plt.title(titles[i])
		plt.imshow(frames[i].reshape(84, 84), 'gray')
	plt.show()

# # maybe i wont use this function
# def frameStacking(observation, lastAction, env, numFrames=4):
# 	# preprocess and stack/concat the frames into frameShape of 1x4x84x84
# 	frames = []
# 	for i in range(numFrames):
# 		img = preprocessing(observation)
# 		frames.append(img)
# 		observation, _, done, _ = env.step(lastAction)
# 		# check done 
# 	return frames
