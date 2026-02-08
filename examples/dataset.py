import os

import torch
import torchvision


class LinMNIST(torch.utils.data.Dataset):
    def __init__(
        self, train, num_templates_per_class, num_samples_per_template, transform=None
    ):
        os.makedirs(os.path.join("examples", "data"), exist_ok=True)
        if train is True:
            dataset = torchvision.datasets.MNIST(
                root=os.path.join("examples", "data"),
                train=True,
                download=True,
            )
        else:
            dataset = torchvision.datasets.MNIST(
                root=os.path.join("examples", "data"),
                train=False,
                download=True,
            )
        data = dataset.data.unsqueeze(1) / 255.0
        targets = dataset.targets
        self.data = list()
        self.targets = list()
        self.transform = transform
        for target in range(10):
            idxs = torch.where(targets == target)[0][:num_templates_per_class]
            self.data = self.data + [data[idxs]]
            self.targets = self.targets + [targets[idxs]]

        self.data = num_samples_per_template * self.data
        self.targets = num_samples_per_template * self.targets

        perm = torch.randperm(len(self.data))
        self.data = torch.cat(self.data, dim=0)[perm]
        self.targets = torch.cat(self.targets, dim=0)[perm]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
