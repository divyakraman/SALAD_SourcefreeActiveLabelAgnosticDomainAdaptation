import numpy as np
import matplotlib.pyplot as plt
import glob
#import imageio
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
from PIL import Image
import os.path as osp
from dataset.cityscapes import cityscapesDataSet
from torch.utils import data
from torch.autograd import Variable
from model.deeplab_multi import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

dtype = torch.cuda.FloatTensor #GPU

data_dir = 'PATH' 
data_list = 'dataset/val.txt' #List of validation images
batch_size = 1
num_steps = 500 #Number of images in data_list
input_size_target = '1024,512' #Size should match size in the train set
eval_set = 'val'
num_workers = 1

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

w, h = map(int, input_size_target.split(','))
input_size_target = (w, h)

targetloader = data.DataLoader(cityscapesDataSet(data_dir, data_list, max_iters=num_steps * batch_size, crop_size=input_size_target, 
    scale=False, mean=IMG_MEAN, set='val'), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

targetloader_iter = enumerate(targetloader)
num_classes = 19

colors = [ [128,64,128],
[244,35,232],
[70,70,70],
[102,102,156],
[190,153,153],
[153,153,153],
[250,170,30],
[220,220,0],
[107,142,35],
[152,251,152],
[70,130,180],
[220,20,60],
[255,0,0],
[0,0,142],
[0,0,70],
[0,60,100],
[0,80,100],
[0,0,230],
[119,11,32] ]
#ignoring void class



def fast_hist(a,b,n):
    k = (a>=0) & (a<n)
    return np.bincount(n*a[k].astype(int)+b[k], minlength=n**2).reshape(n,n)

def per_class_iu(hist):
    return np.diag(hist)/(hist.sum(1)+hist.sum(0)-np.diag(hist))

#Load our trained model
net = torch.load('model.pth', map_location='cpu')

net = net.cuda()

hist = np.zeros((num_classes,num_classes))

for iteration in range(0,num_steps):
    _, batch = targetloader_iter.__next__()
    images, labels, name = batch
    images = Variable(images).cuda()
    pred, a, b = net(images)
            
    pred = pred.detach()
    pred = pred.cpu()
    pred = pred.numpy()
    pred = pred[0,:,:,:]
    pred = np.argmax(pred,0)
    labels = labels.cpu()
    labels = labels.numpy()
    labels = labels[0,:,:]
    labels = labels.astype(np.int)
    hist += fast_hist(labels.flatten(), pred.flatten(), num_classes)

    mIoUs = per_class_iu(hist)
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))

    torch.cuda.empty_cache() #clear cached memory
    print(iteration)

mIoUs = per_class_iu(hist)

for ind_class in range(num_classes):
    print('===> Class '+str(ind_class)+':\t'+str(round(mIoUs[ind_class] * 100, 2)))

print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))

print('===> Accuracy Overall: ' + str(np.diag(hist).sum() / hist.sum() * 100))
acc_percls = np.diag(hist) / (hist.sum(1) + 1e-8) 




