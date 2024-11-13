import os
from torchvision.datasets.vision import VisionDataset
import torch
from PIL import Image
import json

class UCF101:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 train_fraction=0.8,
                 seed=0):
        
        self.train_dataset, self.test_dataset = PytorchUCF101(location, preprocess, "train"), PytorchUCF101(location, preprocess, "test")
       
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        self.classnames = self.train_dataset.class_names
        
class UCF101Val:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 train_fraction=0.8,
                 seed=0):

        self.train_dataset, self.test_dataset = PytorchUCF101(location, preprocess, "train"), PytorchUCF101(location, preprocess, "val")        
       
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        self.classnames = self.train_dataset.class_names


class PytorchUCF101(VisionDataset):

    dataset_dir = 'UCF-101-midframes'

    def __init__(self, root, transform, split="train"):
        self.image_dir = os.path.join(root, self.dataset_dir)
        self.split = os.path.join(self.image_dir, "split_zhou_UCF101.json")
        self.transform = transform
        
        self.files = json.load(open(self.split))[split]

        self.data, self.targets, self.class_names = [], [], []
        for s in self.files:
            self.data.append(os.path.join(self.image_dir, s[0]))
            self.targets.append(s[1])
            if s[2] not in self.class_names:
                self.class_names.append(s[2])

    def __getitem__(self, index):
        image, target = Image.open(self.data[index]), self.targets[index]
        return self.transform(image), target

        
    def __len__(self):
        return len(self.data)
