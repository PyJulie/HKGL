import torch
import numpy as np
import torch.nn as nn

total_epoch = 50
for epoch in range(total_epoch):
    idx =  epoch // 20
    betas = [0, 0.9999, 0.9999]
    effective_num = 1.0 - np.power(betas[idx], cls_num_list)
    per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    criterion = nn.BCELoss(weight = per_cls_weights)