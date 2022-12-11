#Entropy metric: compute prediction probability, and find entropy. 

import torch
import torch.nn.functional as F

def entropy(images,net):
    net.eval()
    pred, a, b = net(images)
    pred = F.softmax(pred)
    pred = pred.detach()
    pred = torch.max(pred[:,:,:],dim=1)
    pred = pred[0]
    
    entropy = -1 * torch.sum(pred * torch.log(pred+1e-30))
    entropy = entropy.detach().cpu().numpy()
    return entropy