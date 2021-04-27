import random
from collections import namedtuple


Experience = namedtuple(
	'Experience',
	('state', 'action', 'reward', 'nextState')
)

class ReplayMemory():
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.pushCount = 0
		
	def push(self, experience):
		if len(self.memory) < self.capacity:
			self.memory.append(experience)
		else:
			self.memory[self.pushCount % self.capacity] = experience
		self.pushCount += 1

	def getMemorySize(self):
		return len(self.memory)

	def sample(self, batchSize):
		return random.sample(self.memory, batchSize)

	def canProvideSample(self, batchSize):
		return batchSize <= len(self.memory)


def main():
	capacity = 4
	batchSize = 2
	mem  = ReplayMemory(capacity)
	exp  = Experience(1, 2, 3, 4)
	exp1 = Experience(5, 6, 7, 8)
	exp2 = Experience(9, 10, 11, 12)
	exp3 = Experience(14, 15, 16, 17)
	exp4 = Experience(18, 19, 20, 21)

	mem.push(exp)
	mem.push(exp1)
	mem.push(exp2)
	mem.push(exp3)
	mem.push(exp4)

	print(mem.canProvideSample(batchSize))
	print(mem.sample(batchSize))

if __name__ == '__main__':
	main()
