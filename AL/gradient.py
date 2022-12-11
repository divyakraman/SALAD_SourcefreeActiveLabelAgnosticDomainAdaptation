#Gradient score
import numpy as np

import torch
from torch import nn
import random
import math
import torch.optim as optim

from scipy import stats

def compute_gradientscore(images, net, pretrained=True):
    
    net.eval()
    #Format of outputs may be different when you use a pretrained model downloaded from a github repo, vs your own model definition; hence using an if statement to simplify
    if(pretrained==True):
        pred, a, b, c = net(images)
    else:
        pred, a, b = net(images)
    #Find the model's hard predictions or pseudo labels
    pred1 = pred.detach()
    pred1 = pred1.cpu()
    pred1 = pred1.numpy()
    pred1 = np.argmax(pred1,1)
    pred1 = torch.from_numpy(pred1)
    pred1 = pred1.cuda()
    #Set optimizer, and set gradients to zero
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer.zero_grad()
    #Criterion for loss computation
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    #Compute loss using the same model's soft and hard predictions
    loss = criterion(pred,pred1)
    #Compute gradients
    loss.backward()
    #Find norm of gradients
    total_norm = 0
    for p in net.parameters():
        try:
            param_norm = p.grad.data.norm(2)
            total_norm = total_norm + param_norm.item() ** 2
        except Exception:
            pass
        
    total_norm = total_norm ** (1. / 2)    

    return total_norm
    
    