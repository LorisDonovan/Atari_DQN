import torch 
import torch.nn as nn

class DQN(nn.Module):
	def __init__(self, num_actions):
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
		self.fc1 = nn.Linear(32*9*9, 256)
		self.out = nn.Linear(256, num_actions)

	def forward(self, x):
		x = torch.relu(self.conv1(x))
		x = torch.relu(self.conv2(x))
		x = x.reshape(-1, 32*9*9)
		x = torch.relu(self.fc1(x))
		x = self.out(x)
		return x


def main():
	net = DQN(4)

	image = torch.rand(32, 4, 84, 84)
	output = net(image)

	print(output.shape)

if __name__ == '__main__':
	main()
