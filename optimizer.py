import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def generate_optimizers(models, lrs, optimizer_type='sgd', weight_decay=0.0005):
    optimizers = []
    if(optimizer_type == 'sgd'):
        for i in range(0, len(models)):
            optimizer = torch.optim.SGD(models[i].parameters(
            ), lr=lrs[i], weight_decay=weight_decay, momentum=0.9)
            optimizers.append(optimizer)

    if(optimizer_type == 'adam'):
        for i in range(0, len(models)):
            optimizer = torch.optim.Adam(models[i].parameters(
            ), lr=lrs[i], weight_decay=weight_decay)
            # optimizer=nn.DataParallel(optimizer)
            optimizers.append(optimizer)
    return optimizers
