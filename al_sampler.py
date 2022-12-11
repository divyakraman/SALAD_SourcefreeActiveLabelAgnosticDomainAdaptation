import argparse
import torch
import torch.nn as nn
import torchvision
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
from AL.classification_entropy import *
from AL.gradientscore import *
from model.deeplab_multi import *
from dataset.target import TargetDataSet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_dir = 'PATH' #Data directory
data_list = 'dataset/unlabeled1.txt' #List of images to sample from
batch_size = 1
num_steps = Length of data_list
input_size_target = '1024,512'
eval_set = 'train'
num_workers = 1

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

w, h = map(int, input_size_target.split(','))
input_size_target = (w, h)

#Dataloader
targetloader = data.DataLoader(TargetDataSet(data_dir, data_list, max_iters=num_steps * batch_size, crop_size=input_size_target, 
    scale=False, mean=IMG_MEAN, set='train'), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

targetloader_iter = enumerate(targetloader)

#Student/ target model network
net = torch.load('model.pth') #Not applicable for round 1 of sampling

#Teacher/source model network
net_pre = torch.load('model_pre.pth') #Not applicable for round 1 of sampling

scores = []
names = []

for iter_num in range(0,num_steps):
    _, batch = targetloader_iter.__next__()
    images, labels, name = batch
            
    images = images.cuda()
    
    score1 = compute_gradientscore(images, net_pre,pretrained=True)
    score2 = entropy(images,net) 
    
    score = score1 * (1/score2) #score = score1 for round 1 because no student model sampling in round 1 
    scores.append(score)
    names.append(name+'\n')
    
args = np.argsort(scores) #ascending order sort

file = open('dataset/roundno.txt','w')
for a in args:
    file.write(names[a])
file.close()
#Pick low score or first n images to sample n images according to the criterion