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
        
        outputs = student_model(inputs)

        soft_student_logits = outputs / temperature

        outputs = sigmoid(outputs)
        soft_student_logits = sigmoid(soft_student_logits)
        
        
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
            soft_teacher_logits = teacher_outputs / temperature
            
        
        soft_teacher_logits = sigmoid(soft_teacher_logits)


        loss = criterion(outputs, labels)
        kd_loss = criterion(soft_student_logits, soft_teacher_logits) # BCE loss or KL loss
        