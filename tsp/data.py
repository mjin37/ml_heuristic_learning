import torch
from torch.utils.data import Dataset

class TSPDataset(Dataset):
    def __init__(self):
        super(TSPDataset, self).__init__()
        self.data_set = []
        self.size = 0

    def random_fill(self, num_nodes, num_samples, random_seed=111):
        torch.manual_seed(random_seed)
        for _ in range(num_samples):
            x = torch.FloatTensor(num_nodes, 2).uniform_(0, 1)
            self.data_set.append(x)

        self.size = len(self.data_set)

    def tensor_fill(self, tensor):
        self.data_set = [tensor]
        self.size = 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx, self.data_set[idx]

