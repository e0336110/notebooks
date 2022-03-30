# -*- coding: utf-8 -*-

### basic modules
import numpy as np
import time, pickle, os, sys, json, PIL, tempfile, warnings, importlib, math, copy, shutil

### torch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F
from torch import autograd

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import argparse

def argparser(data='cifar10', model='large',
              batch_size=128, epochs=200, warmup=10, rampup=121,
              augmentation=True,
              seed=0, verbose=200, 
              epsilon=36/255, epsilon_infty=8/255, epsilon_train=36/255, epsilon_train_infty=8/255, starting_epsilon=0.0, 
              opt='adam', lr=0.001, momentum=0.9, weight_decay=0.0, step_size=10, gamma=0.5, lr_scheduler='step', wd_list=None, 
              starting_kappa=1.0, kappa=0.0,
              niter=100, 
              opt_iter=1, sniter=1, test_opt_iter=1000, test_sniter=1000000): 

    parser = argparse.ArgumentParser()
    
    # main settings
    parser.add_argument('--method', default='BCP')
    parser.add_argument('--rampup', type=int, default=rampup) ## rampup
    parser.add_argument('--warmup', type=int, default=warmup)
    parser.add_argument('--sniter', type=int, default=sniter) ###
    parser.add_argument('--opt_iter', type=int, default=opt_iter) 
    parser.add_argument('--linfty', action='store_true')
    parser.add_argument('--no_save', action='store_true') 
    parser.add_argument('--test_pth', default=None)
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--bce', action='store_true')
    parser.add_argument('--pgd', action='store_true')

    # optimizer settings
    parser.add_argument('--opt', default='adam')
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--step_size", type=int, default=step_size)
    parser.add_argument("--gamma", type=float, default=gamma)
    parser.add_argument("--wd_list", nargs='*', type=int, default=wd_list)
    parser.add_argument("--lr_scheduler", default=lr_scheduler)
    
    # test settings during training
    parser.add_argument('--train_method', default='BCP') 
    parser.add_argument('--test_sniter', type=int, default=test_sniter) 
    parser.add_argument('--test_opt_iter', type=int, default=test_opt_iter) 

    # pgd settings
    parser.add_argument("--epsilon_pgd", type=float, default=epsilon)
    parser.add_argument("--alpha", type=float, default=epsilon/4)
    parser.add_argument("--niter", type=float, default=niter)
    
    # epsilon settings
    parser.add_argument("--epsilon", type=float, default=epsilon)
    parser.add_argument("--epsilon_infty", type=float, default=epsilon_infty)
    parser.add_argument("--epsilon_train", type=float, default=epsilon_train)
    parser.add_argument("--epsilon_train_infty", type=float, default=epsilon_train_infty)
    parser.add_argument("--starting_epsilon", type=float, default=starting_epsilon)
    parser.add_argument('--schedule_length', type=int, default=rampup) ## rampup
    
    # kappa settings
    parser.add_argument("--kappa", type=float, default=kappa)
    parser.add_argument("--starting_kappa", type=float, default=starting_kappa)
    parser.add_argument('--kappa_schedule_length', type=int, default=rampup) ## rampup

    # model arguments
    parser.add_argument('--model', default=model)
    parser.add_argument('--model_factor', type=int, default=8)
    parser.add_argument('--resnet_N', type=int, default=1)
    parser.add_argument('--resnet_factor', type=int, default=1)


    # other arguments
    parser.add_argument('--prefix')
    parser.add_argument('--data', default=data)
    parser.add_argument('--real_time', action='store_true')
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--verbose', type=int, default=200)
    parser.add_argument('--cuda_ids', type=int, default=None)
    
    # loader arguments
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--test_batch_size', type=int, default=batch_size)
    parser.add_argument('--normalization', action='store_true')
    parser.add_argument('--no_augmentation', action='store_true', default=not(augmentation))
    parser.add_argument('--drop_last', action='store_true')
    parser.add_argument('--no_shuffle', action='store_true')

    
    args = parser.parse_args()
    
    args.augmentation = not(args.no_augmentation)
    args.shuffle = not(args.no_shuffle)
    args.save = not(args.no_save)
    
    if args.rampup:
        args.schedule_length = args.rampup
        args.kappa_schedule_length = args.rampup 
    if args.epsilon_train is None:
        args.epsilon_train = args.epsilon 
    if args.epsilon_train_infty is None:
        args.epsilon_train_infty = args.epsilon_infty 
    if args.linfty:
        print('LINFTY TRAINING')
        args.epsilon = args.epsilon_infty
        args.epsilon_train = args.epsilon_train_infty
        args.epsilon_pgd = args.epsilon   
        args.alpha = args.epsilon/4        
        
    if args.starting_epsilon is None:
        args.starting_epsilon = args.epsilon
    if args.prefix:
        args.prefix = 'models/'+args.data+'/'+args.prefix
        if args.model is not None: 
            args.prefix += '_'+args.model

        if args.method is not None: 
            args.prefix += '_'+args.method

        banned = ['verbose', 'prefix',
                  'resume', 'baseline', 'eval', 
                  'method', 'model', 'cuda_ids', 'load', 'real_time', 
                  'test_batch_size', 'augmentation','batch_size','drop_last','normalization',
                  'print','save','step_size','epsilon','gamma','linfty','lr_scheduler',
                  'seed','shuffle','starting_epsilon','kappa','kappa_schedule_length',
                  'test_sniter','test_opt_iter', 'niter','epsilon_pgd','alpha','schedule_length',
                  'epsilon_infty','epsilon_train_infty','test_pth','wd_list','momentum', 'weight_decay',
                  'resnet_N', 'resnet_factor','bce','no_augmentation','no_shuffle','no_save','pgd']
        if args.method == 'baseline':
            banned += ['epsilon', 'starting_epsilon', 'schedule_length', 
                       'l1_test', 'l1_train', 'm', 'l1_proj']

        # if not using a model that uses model_factor, 
        # ignore model_factor
        if args.model not in ['wide', 'deep']: 
            banned += ['model_factor']

        for arg in sorted(vars(args)): 
            if arg not in banned and getattr(args,arg) is not None: 
                args.prefix += '_' + arg + '_' +str(getattr(args, arg))

        if args.schedule_length > args.epochs: 
            raise ValueError('Schedule length for epsilon ({}) is greater than '
                             'number of epochs ({})'.format(args.schedule_length, args.epochs))
    else: 
        args.prefix = 'models/'+args.data+'/temporary'

    if args.cuda_ids is not None: 
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
#         os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids
        # torch.cuda.set_device(args.cuda_ids)


    return args



def select_model(data, m): 
    if data=='mnist':
        if m == 'large': ### Wong et al. large
            model = mnist_model_large()
        elif m == 'large2': ### Wong et al. large
            model = mnist_model_large2()
        else: ### Wong et al. small
            model = mnist_model() 
    elif data=='cifar10':
        if m == 'large':  ### Wong et al. large
            model = cifar_model_large()
        elif m == 'M':  ### CROWN-IBP M
            model = cifar_model_M()
        elif m == 'CIBP':  ### CROWN-IBP
            print('CIBP model')
            model = model_cnn_4layer(3,32,8,512)
        elif m == 'CIBP_noinit':  ### CROWN-IBP
            print('CIBP model no init')
            model = model_cnn_4layer_noinit(3,32,8,512)
        elif m == 'c6f2':
            model = c6f2()
        elif m == 'c6f2_': 
            model = c6f2_()
        else: ### Wong et al. small
            model = cifar_model()
    elif data=='tinyimagenet':
        model = tinyimagenet()
        
        
    return model


def mnist_model(): 
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def mnist_model_large(): 
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*7*7,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model


def mnist_model_large2(): 
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1024,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model


def cifar_model(): 
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model

class Net(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
#             nn.MaxPool2d(2, 2),
            
            nn.Flatten(), 
            nn.Linear(16*(input_shape[1])*(input_shape[2]), 128),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(128, out_channels)
        )
        
    def forward(self, x):
        return self.network(x)

def cifar_model_large(): 
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model


def model_cnn_4layer(in_ch, in_dim, width, linear_size): 
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 4*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8*width, 8*width, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*width*(in_dim // 4)*(in_dim // 4),linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    
    return model




def model_cnn_4layer_noinit(in_ch, in_dim, width, linear_size): 
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 4*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8*width, 8*width, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*width*(in_dim // 4)*(in_dim // 4),linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d):
#             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))
#             m.bias.data.zero_()
    
    return model

def cifar_model_M(): 
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model


def c5f2():
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, stride=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=2),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model


# def c6f2():
#     model = nn.Sequential(
#         nn.Conv2d(3, 32, 3, stride=1, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(32, 32, 3, stride=1, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(32, 32, 4, stride=2, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(32, 64, 3, stride=1, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(64, 64, 3, stride=1, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(64, 64, 4, stride=2),
#         nn.ReLU(),
#         Flatten(),
#         nn.Linear(3136,512),
#         nn.ReLU(),
#         nn.Linear(512,10)
#     )
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d):
#             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))
#             m.bias.data.zero_()
#     return model


def c6f2_():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(4096,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model

def tinyimagenet():
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2),
        nn.ReLU(),
        
        nn.Conv2d(64, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 4, stride=2),
        nn.ReLU(),
        
        nn.Conv2d(128, 256, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, 4, stride=2),
        nn.ReLU(),
        
        Flatten(),
        nn.Linear(9216,256),
        nn.ReLU(),
        nn.Linear(256,200)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model



############################## Flatten / one_hot

class Flatten(nn.Module): ## =nn.Flatten()
    def forward(self, x):
        return x.view(x.size()[0], -1)

    
def one_hot(batch,depth=10):
    ones = torch.eye(depth)
    return ones.index_select(0,batch)


##############################





def train(loader, model, opt, epoch, log, verbose):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    
    model.train()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X, y
        data_time.update(time.time() - end)
 
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.max(1)[1] != y).float().sum()  / X.size(0)
        
        loss = ce
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))
        
        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()


        print(epoch, i, ce.item(), file=log) ########
        if verbose and (i==0 or i==len(loader)-1 or (i+1) % verbose == 0): 
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.4f} ({errors.avg:.4f})'.format(
                   epoch, i+1, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, errors=errors))
        log.flush()
              
              

def evaluate(loader, model, epoch, log, verbose):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.eval()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X, y
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        # print to logfile
        print(epoch, i, ce.item(), err.item(), file=log)

        # measure accuracy and record loss
        losses.update(ce.data, X.size(0))
        errors.update(err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and (i==0 or i==len(loader)-1 or (i+1) % verbose == 0): 
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {error.val:.4f} ({error.avg:.4f})'.format(
                      i+1, len(loader), batch_time=batch_time, loss=losses,
                      error=errors))
        log.flush()

    print(' * Error {error.avg:.4f}'
          .format(error=errors))
    return errors.avg


def pgd_l2(model_eval, X, y, epsilon=36/255, niters=100, alpha=9/255):   
    EPS = 1e-24
    X_pgd = Variable(X.data, requires_grad=True)
    
    for i in range(niters): 
        opt = optim.Adam([X_pgd], lr=1.)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model_eval(X_pgd), y)
        loss.backward()
        grad = 1e10*X_pgd.grad.data
                
        grad_norm = grad.view(grad.shape[0],-1).norm(2, dim=-1, keepdim=True)
        grad_norm = grad_norm.view(grad_norm.shape[0],grad_norm.shape[1],1,1)
                    
        eta = alpha*grad/(grad_norm+EPS)
        eta_norm = eta.view(eta.shape[0],-1).norm(2,dim=-1)
         
        
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = X_pgd.data-X.data                           
        mask = eta.view(eta.shape[0], -1).norm(2, dim=1) <= epsilon
        
        scaling_factor = eta.view(eta.shape[0],-1).norm(2,dim=-1)+EPS
        scaling_factor[mask] = epsilon
        
        eta *= epsilon / (scaling_factor.view(-1, 1, 1, 1)) 

        X_pgd = torch.clamp(X.data + eta, 0, 1)
        X_pgd = Variable(X_pgd.data, requires_grad=True)          
   
    return X_pgd.data
              

def pgd(model_eval, X, y, epsilon=8/255, niters=100, alpha=2/255): 
    X_pgd = Variable(X.data, requires_grad=True)
    for i in range(niters): 
        opt = optim.Adam([X_pgd], lr=1.)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model_eval(X_pgd), y)
        loss.backward()
        eta = alpha*X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        
        X_pgd = torch.clamp(X.data + eta, 0, 1)
        X_pgd = Variable(X_pgd, requires_grad=True)          
       
    return X_pgd.data

def evaluate_pgd(loader, model, args):
    losses = AverageMeter()
    errors = AverageMeter()

    model.eval()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X, y
        if args.linfty:
            X_pgd = pgd(model, X, y, args.epsilon, args.niter, args.alpha)
        else:
            X_pgd = pgd_l2(model, X, y, args.epsilon, args.niter, args.alpha)
            
        out = model(Variable(X_pgd))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)
        losses.update(ce.data, X.size(0))
        errors.update(err, X.size(0))
    print(' * Error {error.avg:.4f}'
          .format(error=errors))
    return errors.avg



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
    
def test(net_eval, test_data_loader, imagenet=0):
    st = time.time()
    n_test = len(test_data_loader.dataset)
    
    err = 0
    n_done = 0
    for j, (batch_images, batch_labels) in enumerate(test_data_loader):

        X = Variable(batch_images)
        Y = Variable(batch_labels)
        out = net_eval(X)
        
        err += (out.max(1)[1].data != (batch_labels-imagenet)).float().sum()
        
        b_size = len(Y)
        n_done += b_size
        acc = 100*(1-err/n_done)
        if j % 10 == 0:
            print('%.2f %%'%(100*(n_done/n_test)), end='\r')

    print('test accuracy: %.4f%%'%(acc))
    
    
def test_topk(net_eval, test_data_loader, k=5, imagenet=1):
    st = time.time()
    n_test = len(test_data_loader.dataset)
    
    err = 0
    n_done = 0
    res = 0
    for j, (batch_images, batch_labels) in enumerate(test_data_loader):

        X = Variable(batch_images)
        Y = Variable(batch_labels)
        out = net_eval(X)
        
        
        b_size = len(Y)
        n_done += b_size
        
        _,pred= out.topk(max((k,)),1,True,True)        
        aa = (batch_labels-imagenet).view(-1, 1).expand_as(pred)
        correct = pred.eq(aa)

        for kk in (k,):
            correct_k = correct[:,:kk].view(-1).float().sum(0)
            res += correct_k# (correct_k.mul_(100.0 / b_size))
            
        
        if j % 10 == 0:
            print('%.2f %%'%(100*(n_done/n_test)), end='\r')

    print('test accuracy: %.4f%%'%(100*res/n_done))
