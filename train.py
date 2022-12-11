import sys
sys.path.append('~/miniconda3/pkgs')
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

from model.deeplab_multi import *
from dataset.TargetDataset import TargetDataSet

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32) #Dataset dependent ; /255.0 works well in most cases

MODEL = 'DeepLab' #Not necessary
BATCH_SIZE = 7
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = 'path to dataset'
DATA_LIST_PATH = 'path to list of images for supervised training'
DATA_LIST_PATH_UNSUP = 'path to list of images for unsupervised training'

INPUT_SIZE = '1024,512' #Desired input image size (After taking into account relevant resizing operations)
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19 #May not be needed depending on model definition; model we are currently using already has it
NUM_STEPS = 50000 #Change depending on task, dataset size
NUM_STEPS_STOP = 50000
POWER = 0.9
RANDOM_SEED = 1234
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1
SNAPSHOT_DIR = './snapshots/model/' #Folder to save trained model in
WEIGHT_DECAY = 0.0005


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    #parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
    #                    help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")


    return parser.parse_args()


args = get_arguments()



def loss_calc(pred, label): #Function to compute supervised loss - task dependent 
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    #
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    label = Variable(torch.from_numpy(label).long()).cuda()
    return ce_loss(pred, label.cuda())

def lr_poly(base_lr, iter, max_iter, power): #Learning rate update
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter): #Update learning rate in network parameters
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def main():
    """Create the model and start the training."""

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    
    cudnn.enabled = True
    
    
    model = DeepLabMulti() #Network initialization; DeepLab for segmentation; resnet for classification; 
    
    model.train()
    
    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    #Supervised training subset of images mined by active learning
    trainloader = data.DataLoader(
        cityscapesDataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN, set='train'),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    trainloader_iter = enumerate(trainloader)
    
    #Unsup loader: all other images
    unsuploader = data.DataLoader(
        cityscapesDataSet(args.data_dir, DATA_LIST_PATH_UNSUP, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN, set='train'),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    unsuploader_iter = enumerate(unsuploader)

    #End of unsup loader 

    
    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    
    interp_target = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear') #Segmentation

    for i_iter in range(0, args.num_steps):

        loss_seg_value2 = 0
        
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        
        for sub_i in range(args.iter_size):

            #Supervised set training
            _, batch = trainloader_iter.__next__()
            images, labels, name = batch
            labels = labels.numpy()
            images = Variable(images).cuda()
            out, distil_loss, pseudo_loss = model(images)
            sup_loss = loss_calc(out,labels)
            loss = sup_loss + 0.1 * distil_loss
            
            #Unsupervised set training
            _, batch = unsuploader_iter.__next__()
            images, labels, name = batch
            labels = labels.numpy()
            images = Variable(images).cuda()
            out, distil_loss, pseudo_loss = model(images)
            loss = loss + 0.1 * distil_loss

            #loss = loss + pseudo_loss #Only for classification where quality of pseudo labels is good. Degrades performance in cases like segmentation where quality of pseudo labels is not good. 
            
            loss = loss.sum()/args.batch_size
            # proper normalization
            loss.backward()
            loss_value = loss.data.cpu().numpy() / args.iter_size

            torch.cuda.empty_cache()

            optimizer.step()
        
        torch.cuda.empty_cache()
        
        print('exp = {}'.format(args.snapshot_dir))
        print('Iteration: ', i_iter, ' Loss: ', loss_value)
       # 'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}'.format(
       #     i_iter, args.num_steps, loss_value))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model, osp.join(args.snapshot_dir, 'CS_' + str(args.num_steps_stop) + '.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model, osp.join(args.snapshot_dir, 'CS_' + str(args.num_steps_stop) + '.pth'))
            
    
            

if __name__ == '__main__':
    main()




