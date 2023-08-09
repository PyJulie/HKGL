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
if use_gpu:
    model_ft = model_ft.cuda()

teacher_model = torch.load('') # load teacher model here.

sigmoid = nn.Sigmoid()
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(student_model.parameters(), lr=1e-3)

for name, module in teacher_model._modules.items():
    for p in module.parameters():
        p.requires_grad = False


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
        
        # forward
        f_teacher = teacher_model.foward_features(inputs)
        f_student = student_model.foward_features(inputs)

        cos_sim = cos(f_teacher, f_student)
        