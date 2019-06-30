import torch
import torch.nn as nn
import numpy as np
from .net.resnet_cifar import ResNets
from torch.autograd import Variable
from utils import to_var, accuracy
import torch.backends.cudnn as cudnn
import os

class model():
    def __init__(self, args):
        super(model, self).__init__()

        self.args = args
        assert torch.cuda.is_available()

        ## Define neural networks...
        self.net = nn.DataParallel(ResNets(args).cuda())
            
        cudnn.benchmark = True
        
        ## ...with their losses...
        self.get_loss = nn.CrossEntropyLoss()
        
        ## ... and optimizers
        self.optimizer = torch.optim.SGD(self.net.parameters(), 
                                        lr=args.lr,
                                        momentum=0.9, 
                                        weight_decay=5e-4)
        self.lr = args.lr
    
    def forward(self, x, label, need_result=False):
        x = self.net(to_var(x))

        label = to_var(label)
        self.loss = self.get_loss(x, label)
        acc1, acc5 = accuracy(x, label, (1,5))
        return self.loss, acc1, acc5
    
    def optimize(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def update_optimizer(self, n_epoch):
        if self.args.dataset == 'tiny_imagenet':
            if (n_epoch - 30) % 30 == 0 and n_epoch > 0:
                lr = self.lr / 10
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print('update learning rate: %f -> %f' % (self.lr, lr))
                self.lr = lr
        else:
            if (n_epoch - 150) % 100 == 0 and n_epoch > 50:
                lr = self.lr / 10
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print('update learning rate: %f -> %f' % (self.lr, lr))
                self.lr = lr
    
    def load(self, addr):
        ckpt = torch.load(addr)
        self.net.load_state_dict(ckpt['net'])
        max_acc = ckpt['max_acc']
        try:
            max_accf = ckpt['max_accf']
        except:
            max_accf = 0
        epoch_start = ckpt['n_epoch'] + 1
        return max_acc, max_accf, epoch_start


    def save(self, ckpt_addr, n_epoch, max_acc, max_accf, save_best=False):
        state = {
            'net': self.net.state_dict(),
            'max_acc': max_acc,
            'max_accf': max_accf,
            'n_epoch': n_epoch
        }
        torch.save(state, os.path.join(ckpt_addr, "last.pth"))
        if save_best:
            torch.save(state, os.path.join(ckpt_addr, "best.pth"))

