import torch
import torch.nn as nn

class ResNet(nn.Module):
	def __init__(self, in_size, hidden_size, out_size, num_hidden_layers):
		super(ResNet, self).__init__()
		self.in_lin = nn.Linear(in_size, hidden_size)
		self.out_lin = nn.Linear(hidden_size, out_size)
		hidden_layers = []
		hidden_norms = []
		for _ in range(num_hidden_layers):
			hidden_layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
			hidden_norms.append(torch.nn.LayerNorm(hidden_size))
		self.hidden_layers = hidden_layers
		self.hidden_norms = hidden_norms

		self.activation = torch.nn.Tanh()

	def forward(self, x):
		x = self.activation(self.in_lin(x))

		for f, norm in zip(self.hidden_layers, self.hidden_norms):
			x = x + self.activation(norm(f(x)))

		return self.out_lin(x)
	
def make_simple_nn(in_size, hidden_size, out_size, hidden_layers):
	activation = torch.nn.Tanh()

	if hidden_layers == 0:
		return torch.nn.Sequential(torch.nn.Linear(in_size, out_size))
	
	model = torch.nn.Sequential()
	model.append(torch.nn.Linear(in_size, hidden_size))
	model.append(activation)
	for _ in range(hidden_layers):
		model.append(torch.nn.Linear(hidden_size, hidden_size))
		model.append(activation)
	model.append(torch.nn.Linear(hidden_size, out_size))

	return model
