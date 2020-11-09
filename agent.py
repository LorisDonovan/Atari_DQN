import math
import random
import torch


class Agent():
	def __init__(self, start, end, decay, numActions, device):
		self.start = start
		self.end = end
		self.decay = decay
		self.numActions = numActions
		self.device = device
		self.currentStep = 0

	def __getExplorationRate(self):
		return self.end + (self.start - self.end) * math.exp(-1 * self.decay * self.currentStep)
	
	def selectAction(self, state, policyNet):
		rate = self.__getExplorationRate()
		print("exploration rate = ", rate)

		self.currentStep += 1
		if rate > random.random():
			return torch.randint(0, numActions, (1,1)).to(self.device)
		else:
			with torch.no_grad():
				return policyNet(state).argmax(dim=1).reshape(-1, 1).to(self.device)


def main():
	policyNet = torch.nn.Sequential(
		torch.nn.Linear(1, 4)
	)
	state = torch.rand([1, 1])
	agent = Agent(0.2, 0.01, 0.01, 4, "cpu")
	action = agent.selectAction(state, policyNet)
	print(action.shape)

if __name__ == '__main__':
	main()
