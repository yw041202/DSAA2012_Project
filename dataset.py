import torch
import numpy as np
from torch.utils.data import Dataset


class RedditDataset(Dataset):
    def __init__(self, labels, tweets, days=200):
        super().__init__()
        self.labels = labels
        self.tweets = tweets
        self.days = days

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        labels = torch.tensor(np.array(self.labels[item]), dtype=torch.long)
        if self.days > len(self.tweets[item]):
            tweets = torch.tensor(np.array(self.tweets[item]), dtype=torch.float32)
        else:
            tweets = torch.tensor(np.array(self.tweets[item][:self.days]), dtype=torch.float32)

        return [labels, tweets]