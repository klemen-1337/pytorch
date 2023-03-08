import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import multiprocessing

"""
DataLoaderji in dataseti so namenjeni nalaganju ter procesiranju podatkov. Omogočajo nam, da razdelimo podatke
na segmente, jih zmešamo, paraleliziramo in naredimo custom razrede za njihove prikaze.
"""
class WineDataset(Dataset):

    def __init__(self):
        #data loading
        xy = np.loadtxt('wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    

def main():
    dataset = WineDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

    #training loop

    num_epochs = 2
    total_samples = len(dataset)
    num_iterations = math.ceil(total_samples/4)

    for epoch in range(num_epochs):

        for i, (inputs, labels) in enumerate(dataloader):
            if (i+1) % 5 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{num_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')



if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

