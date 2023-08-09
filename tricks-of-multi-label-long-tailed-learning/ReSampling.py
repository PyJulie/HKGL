import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms
import numpy as np
import time
import os
import torch.nn.functional as F

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        #label_to_count = [0] * len(np.unique(dataset.img_label))
        label_to_count = np.zeros(29,dtype=int)
        
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[np.where(label)] += 1
            #print(label, label_to_count)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        target = dataset.img_label
        print(per_cls_weights)
        # weight for each sample
        weights = [per_cls_weights[np.where(target[idx])[0][-1]] for idx in range(len(target))]
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.img_label[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples