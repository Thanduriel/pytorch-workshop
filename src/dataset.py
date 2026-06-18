from torch.utils.data import Dataset
import torch

class BasicDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.Tensor(inputs).to(torch.float32)
        if self.inputs.dim() == 1:
            self.inputs = self.inputs.unsqueeze(1)

        self.targets = torch.Tensor(targets).to(torch.float32)
        if self.targets.dim() == 1:
            self.targets = self.targets.unsqueeze(1)
        
        assert(self.inputs.size(0) == self.targets.size(0))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
