import gym
from gym.utils.play import play

import cv2 
import random 
import numpy as np 
import matplotlib.pyplot as plt
from itertools import count

import torch 
import torch.optim as optim

from dqn import DQN
from replayMemory import ReplayMemory, Experience
from agent import Agent
from utils import *


# Hyperparameters
EPISODES = 100
TARGET_UPDATE = 1000	# target net update frequency
LEARNIN_RATE = 0.001

EP_START = 1.0		# epsilon start
EP_END = 0.1 		# epsilon end
EP_DECAY = 0.0001	# epsilon decay

MEM_CAPACITY = 64	# replay memory capacity
BATCH_SIZE = 4


def playGame(name = "BreakoutNoFrameskip-v4"):
	env = gym.make(name)
	play(env, zoom=4) # to play the environment

def testMain():
	device = torch.device("cpu")

	env = gym.make("BreakoutDeterministic-v4") # Deterministic-v4; frameskip = 4
	# play(env, zoom=4) # to play the environment
	
	numActions = env.action_space.n
	mem = ReplayMemory(MEM_CAPACITY)
	agent = Agent(EP_START, EP_END, EP_DECAY, numActions, device)
	policyNet = DQN(numActions).to(device)
	targetNet = DQN(numActions).to(device)
	targetNet.load_state_dict(policyNet.state_dict())
	targetNet.eval()

	obv = env.reset()
	lastAction = 0
	frames = frameStacking(obv, lastAction, env)
	plotSubplot(4, 1, 4, frames)

def algorithmTest():
	device = torch.device("cpu")
	env = gym.make("BreakoutDeterministic-v4") # Deterministic-v4; frameskip = 4
	# play(env, zoom=4) # to play the environment
	
	numActions = env.action_space.n
	mem = ReplayMemory(MEM_CAPACITY)
	agent = Agent(EP_START, EP_END, EP_DECAY, numActions, device)
	policyNet = DQN(numActions).to(device)
	targetNet = DQN(numActions).to(device)
	targetNet.load_state_dict(policyNet.state_dict())
	targetNet.eval()
	optimizer = optim.Adam(params=policyNet.parameters(), lr=LEARNIN_RATE)

	for ep in range(EPISODES):
		print('episode: ', ep+1)
		done = False 
		obv = env.reset()
		preproObv = preprocessing(obv)
		frames = [preproObv]
		nextFrames = []
		lastAction = 0
		totalReward = 0

		for t in count():
			if len(frames) == 4:
				state = torch.cat(frames, dim=1) # returns tensor of 1x4x84x84
				action = agent.selectAction(state, policyNet)
				frames = []
			else:
				action = lastAction

			obv, reward, done, _ = env.step(action)
			preproObv = preprocessing(obv)
			frames.append(preproObv)
			nextFrames.append(preproObv)
			totalReward += reward # for evalution
			if done:
				reward = -1.0

			if len(nextFrames) == 4:
				nextState = torch.cat(nextFrames, dim=1) # returns tensor of 1x4x84x84
				nextFrames = []

			lastAction = action
			env.render()
			if done:
				break


def main():
	# playGame()
	algorithmTest()

if __name__ == '__main__':
	main()


'''
# nextState how???
exp = Experience(state, action, reward, nextState)
# state = nextState
mem.push(exp)
if mem.canProvideSample(BATCH_SIZE):
	exp = mem.sample(BATCH_SIZE)
	# extract to tensors
	state, action, reward, nextState = extractTensors(exp)
	q_pred = policyNet(state).gather(1, action)

	q_target = targetNet(nextState).max(dim=1, keep_dim=True)
	target = GAMMA * q_target + reward

	loss = torch.smooth_l1_loss(q_pred, target)
	policyNet.zero_grad()
	loss.backward()
	optimizer.step()

if t % TARGET_UPDATE == 0:
	targetNet.load_state_dict(policyNet.state_dict())
'''



