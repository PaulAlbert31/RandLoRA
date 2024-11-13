"""
MNIST data utilities

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning

Modified from the codebase by Ilharco et al.,
at https://github.com/mlfoundations/task_vectors
"""

import os
import json
import torch
import torchvision.datasets as datasets

class MNIST:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16):


        self.train_dataset = datasets.MNIST(
            root=location,
            download=True,
            train=True,
            transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset = datasets.MNIST(
            root=location,
            download=True,
            train=False,
            transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        self.class_splits = self.load_class_splits(location)

    def split_class_data(self, train):
        """Find indices of the data corresponding to each class

        Parameters:
        -----------
        train: bool
            If True, find indices from the training set. Otherwise, find indices
            from the test set.

        Returns:
        --------
        indices: dict
            A dictionary with class index as the key, and the list of data indices
            as the value.
        """
        indices = {}
        dataset = self.train_dataset if train else self.test_dataset
        for i, (_, t) in enumerate(dataset):
            if t not in indices:
                indices[t] = [i,]
            else:
                indices[t].append(i)
        return indices

    def load_class_splits(self, location):
        """Load the list of data indices for each class"""

        root_dir = os.path.join(location, self.__class__.__name__)
        cache_path = os.path.join(root_dir, 'class_splits.json')
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                class_splits = json.load(f)
            return class_splits
        else:
            print(
                f"Class splits for {self.__class__.__name__} not found."
                "\nGenerating and caching class splits..."
            )
            class_splits = {
                'train': self.split_class_data(True),
                'test': self.split_class_data(False),
            }
            with open(cache_path, 'w') as f:
                json.dump(class_splits, f)
            return class_splits