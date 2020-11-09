import gym
from gym.utils.play import play

import cv2 
import random 
import numpy as np 
import matplotlib.pyplot as plt
from itertools import count

import torch 
import torch.optim as optim
import torch.nn.functional as functional

from dqn import DQN
from replayMemory import ReplayMemory, Experience
from agent import Agent
from utils import *


# Hyperparameters
EPISODES = 1000
TARGET_UPDATE = 5000 # target net update frequency
LEARNIN_RATE = 0.001
GAMMA = 0.95

EP_START = 0.99   # epsilon start
EP_END = 0.1      # epsilon end
EP_DECAY = 0.0001 # epsilon decay

MEM_CAPACITY = 32*1024 # replay memory capacity
BATCH_SIZE = 32


def playGame(name = "BreakoutNoFrameskip-v4"):
	env = gym.make(name)
	play(env, zoom=4) # to play the environment


def evaluation():
	device = torch.device("cuda")
	env = gym.make("BreakoutDeterministic-v4") # Deterministic-v4; frameskip = 4

	numActions = env.action_space.n
	policyNet = DQN(numActions)
	# policyNet = torch.load("SavedModels/Policy.pt")
	policyNet.to(device)

	for ep in range(EPISODES):
		print('episode: ', ep+1)
		done = False 
		obv = env.reset()
		preproObv = preprocessing(obv)
		frames = [preproObv]
		lastAction = torch.zeros(1,1).to(device)

		for _ in count():
			if len(frames) == 4:
				state = torch.cat(frames, dim=1).to(device) # returns tensor of 1x4x84x84
				with torch.no_grad():
					action = policyNet(state).argmax(dim=1).reshape(-1, 1).to(device)
				frames = []
			else:
				action = lastAction
			
			obv, _, done, _ = env.step(action)
			preproObv = preprocessing(obv)
			frames.append(preproObv)
		
			lastAction = action
			env.render()
			if done:
				break

	print("Completed!!!")


def algorithmImpl():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

	stepCount = 0

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
				state = torch.cat(frames, dim=1).to(device) # returns tensor of 1x4x84x84
				action = agent.selectAction(state, policyNet)
				frames = []
			else:
				action = lastAction

			obv, r, done, _ = env.step(action)
			preproObv = preprocessing(obv)
			frames.append(preproObv)
			nextFrames.append(preproObv)
			totalReward += r # for evalution
			if done:
				r = -1.0
			reward = torch.tensor(r).reshape(1,1).to(device)

			lastAction = action

			if len(nextFrames) == 4:
				nextState = torch.cat(nextFrames, dim=1).to(device) # returns tensor of 1x4x84x84
				nextFrames = []
				mem.push(Experience(state, action, reward, nextState))
				state = nextState
				
				if mem.canProvideSample(BATCH_SIZE):
					exps = mem.sample(BATCH_SIZE)
					states, actions, rewards, nextStates = extractTensors(exps)
					qPred = policyNet(states).gather(1, actions)

					qTarget = targetNet(nextStates).max(dim=1, keepdim=True)[0].detach()
					target = GAMMA * qTarget + rewards

					loss = functional.mse_loss(qPred, target)
					policyNet.zero_grad()
					loss.backward()
					optimizer.step()
				
				stepCount += 1
				if stepCount == TARGET_UPDATE:
					stepCount = 0
					targetNet.load_state_dict(policyNet.state_dict())
					print("SavedModels/Saved model")
					torch.save(policyNet, "Policy.pt")
			
			if ep % 15 == 0:
				env.render()

			if done:
				break
	torch.save(policyNet, "SavedModels/Policy.pt")


def main():
	# playGame()
	# algorithmImpl()
	evaluation()

if __name__ == '__main__':
	main()




