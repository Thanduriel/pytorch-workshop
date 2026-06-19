from poisson import generate_data
from train import train
import neuralnets

import matplotlib.pyplot as plt
import torch

plot_dataset = True
plot_losses = True
plot_eval = True

default_cycler = plt.cycler(color=['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE',
                    '#AA3377', '#BBBBBB', '#000000'],
					marker=['o', 's', '*', 'v', '^', 'D', '+', 'x'])

plt.rcParams.update({
    'axes.prop_cycle': default_cycler
})

def plot_solution(grid, fs, us, labels):
	if fs is None:
		for u, label in zip(us, labels):
			plt.plot(grid, u, label=f'u {label}')
	else:
		for f, u, label in zip(fs, us, labels):
			plt.plot(grid, f, label=f'f {label}', linestyle='--')
			plt.plot(grid, u, label=f'u {label}')
	plt.legend()
	plt.show()

seed = 17175383

# data generation
inputs_train, targets_train, grid = generate_data(32, 128, k_min=1, k_max=8, seed=seed, f_noise_std=0.1, amp_decay=2.0) # 2.0, 4.0
inputs_valid, targets_valid, _ = generate_data(32, 16, k_min=1, k_max=8, seed=seed, f_noise_std=0.1, amp_decay=0.5) # 0.5, 0.25

if plot_dataset:
	plot_solution(grid, [inputs_train[0], inputs_valid[0]], [targets_train[0], targets_valid[0]], labels=['train', 'valid'])

# training
torch.manual_seed(seed)
nets = []
wd_settings = [0.0, 0.1, 0.5]
for wd in wd_settings: # , 0.1, 0.5
	net = neuralnets.ResNet(inputs_train.shape[1], 80, targets_train.shape[1], 6)
	train_losses, valid_losses = train(net, (inputs_train, targets_train), (inputs_valid, targets_valid), 128, lr=0.001, weight_decay=wd)
	nets.append(net)
	if plot_losses:
		plt.semilogy(train_losses, label=f'train wd={wd}', markevery=8)
		plt.semilogy(valid_losses, label=f'validation wd={wd}', linestyle='--', markevery=8)

if plot_losses:
	plt.legend()
	plt.show()

# evaluation
if plot_eval:
	net.eval()
	with torch.inference_mode():
		net_outputs = []
		for net in nets:
			outputs_valid = net(torch.Tensor(inputs_valid).to(torch.float32))
			net_outputs.append(outputs_valid[0])

		plot_solution(grid, None, [targets_valid[0]] + net_outputs, labels=['ref'] + wd_settings)
