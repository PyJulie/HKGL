import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms
import numpy as np
import time
import os
from torch.utils.data import Dataset
from PIL import Image
import argparse

import torchvision

student_model = '' # define student model here.
teacher_model = torch.load('') # load teacher model here.

# load the map dic here, which could be like:
# {'Normal': [0],
#  'DR': [2, 3, 10, 11],
#  'Cataract': [4],
#  'Glaucoma': [5],
#  'AMD': [7, 9],
#  'Hyper': [8],
#  'Myopia': [6],
#  'Others': [1]}

dic_super_map_verse = np.load('',allow_pickle = True).item()

sigmoid = nn.Sigmoid()

criterion = nn.BCELoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(student_model.parameters(), lr=1e-3)

for name, module in teacher_model._modules.items():
    for p in module.parameters():
        p.requires_grad = False

temperature = 10 # define temperature value here.
total_epoch = 50
for i in range(total_epoch):
    student_model.train()
    for data in tqdm.tqdm(train_dataloaders):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer_ft.zero_grad()
        
        outputs = model_ft(inputs)
        super_outputs = torch.Tensor().cuda()
        super_labels = torch.Tensor().cuda()
        #print(labels)
        for _idx in range(len(keys)):
            key = keys[_idx]
            coarse_idx = dic_super_map_verse[key]

            tmp_outputs = outputs[:,coarse_idx]
            tmp_outputs = torch.sum(tmp_outputs,dim=1).view(len(tmp_outputs),-1)
            super_outputs = torch.cat([super_outputs,tmp_outputs],dim=1)
            
            coarse_idx = dic_super_map_verse[key]

            tmp_labels = labels[:,coarse_idx]
            tmp_labels = torch.sum(tmp_labels,dim=1).view(len(tmp_labels),-1)
            super_labels = torch.cat([super_labels,tmp_labels],dim=1)

        tmp_labels = tmp_labels.cuda()
        tmp_outputs = tmp_outputs.cuda()

        super_outputs = sigmoid(super_outputs)
        loss_super = criterion(super_outputs,super_labels)
        
        outputs = sigmoid(outputs)
        loss_coarse = criterion(outputs, labels)
        loss = loss_super + loss_coarse
        