import torch
from torch.utils.data import DataLoader
from dataset import BasicDataset

def train(net, train_data, valid_data, epochs, lr=0.001, weight_decay=0.0):
	train_losses = []
	valid_losses = []
	
	return train_losses, valid_losses
