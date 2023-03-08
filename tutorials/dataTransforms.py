import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import multiprocessing

"""
Transformi se uporabljajo za PIL slike, tenzorje, ndarraye, custom podatke ob ustvarjanju dataseta.
Namen je transformirati podatke v določeno željeno obliko.
Lahko uporabljamo custom transformerje ali pa že nekatere od obstoječih:

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale
On Tensors
----------
LinearTransformation, Normalize, RandomErasing
Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage
Generic
-------
Use Lambda 
Custom
------
Write own class


"""

class WineDataset(Dataset):

    def __init__(self, transform=None):
        #data loading
        xy = np.loadtxt('wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        self.x = xy[:, 1:]
        self.y = xy[:, [0]] # n_samples, 1

        self.transform=transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    
class MulTransform:
    def __init__(self,factor):
        self.factor = factor
    
    def __call__(self,sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets


dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

