import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from lib.net.resnet import ResNet101
import itertools
import datetime

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=16, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--fix-epoch', default=10, type=int, metavar='FN',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--dataset', type=str, default='tiny-imagenet-200',\
                    help='dataset to use for this training.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str,
                    help='addr pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--norm', default='Dynamic', type=str, help='norm use in model')

parser.add_argument('--freeze-norm', action='store_true')


best_prec1 = 0
best_prec5 = 0
ckpt_dir = ""
ckpt_addr = ""
my_weight = 0

def main():
    global args, best_prec1, best_prec5, ckpt_addr, ckpt_dir
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch]()
    # ckpt_dir = datetime.datetime.now().strftime("%m%d%H%M%S") + "_imageNet_{}".format(args.norm)
    # ckpt_addr = os.path.join('ckpt', ckpt_dir)
    # os.mkdir(ckpt_addr)

    # # model, norm = ResNet101(args)
    # # for n, p in model.named_parameters():
    # #     print(n)
    # # # print(n for n, p in model.named_parameters())
    # # exit()

    # if not args.distributed:
    #     if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #         model.features = torch.nn.DataParallel(model.features)
    #         model.cuda()
    #     else:
    #         model = torch.nn.DataParallel(model).cuda()
    # else:
    #     model.cuda()
    #     model = torch.nn.parallel.DistributedDataParallel(model)
    
    # if args.pretrained:
    #     print('Loading pretrained weight...')
    #     load_pretrained(model, args.pretrained)
    #     print('Done.')

    # # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()

    # dn_params = (p for p in [p for n, p in model.named_parameters() if 'mn' in n])
    # other_params = (p for p in [p for n, p in model.named_parameters() if 'mn' not in n])

    # optimizer = torch.optim.SGD([{'params': dn_params}, 
    #                              {'params': other_params, 'lr': 0}], args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    
    # optimizer_2 = torch.optim.SGD(model.parameters(), args.lr, 
    #                               momentum=args.momentum,
    #                               weight_decay=args.weight_decay)

    # # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         best_prec1 = checkpoint['best_prec1']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ]))
    
    print(get_mean_and_std(train_dataset))
    exit()

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        if args.freeze_norm:
            if epoch - args.start_epoch <= args.fix_epoch:
                train(train_loader, model, criterion, optimizer, epoch)
            else:
                train(train_loader, model, criterion, optimizer_2, epoch)
        else:
            train(train_loader, model, criterion, optimizer_2, epoch)

        # evaluate on validation set
        with torch.no_grad():
            prec1, prec5 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        best_prec5 = max(prec5, best_prec5)
        print("Best acc@1: {}, acc@5: {}".format(best_prec1, best_prec5))
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    global my_weight
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        my_weight = (epoch * len(train_loader) + i) / (args.fix_epoch * len(train_loader)) 
        if my_weight > 1:
            my_weight = 1

        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var, my_weight)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('{4} Training: \n'
                  'Epoch: [{0}][{1}/{2}]\n'
                  'Time {batch_time.val:.4f} ({batch_time.avg:.4f}) / {3}\n'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\n'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                  'Prec@1 {top1.val:.4f} ({top1.avg:.4f})\n'
                  'Prec@5 {top5.val:.4f} ({top5.avg:.4f})\n'.format(
                   epoch, i, len(train_loader), 
                   GetTime(((args.epochs-epoch) * len(train_loader) - i) * batch_time.avg), 
                   ckpt_dir,
                   batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var, my_weight)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('{4} Validating: \n'
                  'Epoch: [{0}][{1}/{2}]\n'
                  'Time {batch_time.val:.4f} ({batch_time.avg:.4f}) / {3}\n'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                  'Prec@1 {top1.val:.4f} ({top1.avg:.4f})\n'
                  'Prec@5 {top5.val:.4f} ({top5.avg:.4f})\n'.format(
                   epoch, i, len(val_loader), 
                   GetTime(((args.epochs-epoch) * len(val_loader) - i) * batch_time.avg), 
                   ckpt_dir, 
                   batch_time=batch_time,
                   loss=losses, top1=top1, top5=top5))


    print(' * Prec@1 {top1.avg:.4f} Prec@5 {top5.avg:.4f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = os.path.join(ckpt_addr, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(ckpt_addr, 'model_best.pth.tar'))


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 4:
        lr = args.lr * (0.1 ** (epoch // 2))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def load_pretrained(net, weight_file):
    pretrained_weight = torch.load(weight_file)
    params = net.state_dict()
    for p_name, p_tensor in pretrained_weight.items():
        d_name_2 = None
        if 'bn' not in p_name and 'downsample.1' not in p_name:
            d_name = p_name
        elif 'weight' in p_name:
            d_name = p_name[: p_name.rfind('.')+1] + 'mn.weight_bias'
            d_name_2 = p_name[: p_name.rfind('.')+1] + 'bn.weight'
        elif 'bias' in p_name:
            d_name = p_name[: p_name.rfind('.')+1] + 'mn.bias_bias'
            d_name_2 = p_name[: p_name.rfind('.')+1] + 'bn.bias'
        else:
            d_name = p_name[: p_name.rfind('.')+1] + 'bn.' + p_name[p_name.rfind('.')+1:]
            # print(p_name)
            # continue

        params[d_name].copy_(p_tensor.view(params[d_name].size()))
        if d_name_2:
            params[d_name_2].copy_(p_tensor.view(params[d_name_2].size()))
    # exit()

def GetTime(seconds):
    t = int(seconds)

    day = t//86400
    hour = (t-(day*86400))//3600
    minit = (t - ((day*86400) + (hour*3600)))//60
    seconds = t - ((day*86400) + (hour*3600) + (minit*60))
    return "{} days {} hours {} minutes {} seconds remaining.".format(day, hour, minit, seconds)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

if __name__ == '__main__':
    main()
