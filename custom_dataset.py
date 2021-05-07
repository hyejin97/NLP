from torch.utils.data import Dataset

class GenderDataset(Dataset):
    def __init__(self, dataset):
        super(DatasetDict, self).__init__()
        self.dataset = dataset
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.dataset[idx]