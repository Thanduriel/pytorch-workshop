import torch
from torch.utils.data import DataLoader
from dataset import BasicDataset

def train(net, train_data, valid_data, epochs, lr=0.001, weight_decay=0.0):
	train_dataset = BasicDataset(train_data[0], train_data[1])
	valid_dataset = BasicDataset(valid_data[0], valid_data[1])
	train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
	valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

	optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
	loss_fn = torch.nn.MSELoss()

	train_losses = []
	valid_losses = []
	for epoch in range(epochs):
		train_loss = 0.0
		train_batches = 0
		for inputs, targets in train_dataloader:
			optimizer.zero_grad()

			outputs = net(inputs)
			loss = loss_fn(outputs, targets)
			loss.backward()
			optimizer.step()

			train_loss += loss.item()
			train_batches += 1

		valid_loss = 0.0
		valid_batches = 0
		for inputs, targets in valid_dataloader:
			outputs = net(inputs)
			loss = loss_fn(outputs, targets)
			
			valid_loss += loss.item()
			valid_batches += 1

		train_loss /= train_batches
		valid_loss /= valid_batches
		train_losses.append(train_loss)
		valid_losses.append(valid_loss)
		print(f"{epoch: 3d}: train: {train_loss}, valid: {valid_loss}")

	return train_losses, valid_losses
