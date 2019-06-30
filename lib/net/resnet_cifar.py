'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.nn.ilm_gn import ilm_gn
from lib.nn.ilm_ln import ilm_ln
from lib.nn.ilm_in import ilm_in

from torch.nn import BatchNorm2d as batch

import lib.nn as mynn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, norm, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                norm(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, norm, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = norm(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                norm(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, norm, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], norm, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], norm, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], norm, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], norm, stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, norm, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, norm, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(args):
    norm = eval(args.norm)
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'imageNet':
        num_classes = 1000
    elif args.dataset == 'tiny_imagenet':
        num_classes = 200
    else:
        raise NotImplementedError
    return ResNet(BasicBlock, [2,2,2,2], norm, num_classes=num_classes)

def ResNet34(args):
    norm = eval(args.norm)
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'imageNet':
        num_classes = 1000
    elif args.dataset == 'tiny_imagenet':
        num_classes = 200
    else:
        raise NotImplementedError
    return ResNet(BasicBlock, [3,4,6,3], norm, num_classes=num_classes)

def ResNet50(args):
    norm = eval(args.norm)
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'imageNet':
        num_classes = 1000
    elif args.dataset == 'tiny_imagenet':
        num_classes = 200
    else:
        raise NotImplementedError
    return ResNet(BasicBlock, [3,4,6,3], norm, num_classes=num_classes)

def ResNet101(args):
    norm = eval(args.norm)
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'imageNet':
        num_classes = 1000
    elif args.dataset == 'tiny_imagenet':
        num_classes = 200
    else:
        raise NotImplementedError
    return ResNet(Bottleneck, [3,4,23,3], norm, num_classes=num_classes)

def ResNet152(args):
    norm = eval(args.norm)
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'imageNet':
        num_classes = 1000
    elif args.dataset == 'tiny_imagenet':
        num_classes = 200
    else:
        raise NotImplementedError
    return ResNet(BasicBlock, [3,8,36,3], norm, num_classes=num_classes)

def ResNets(args):
    if args.resnet == 101:
        return ResNet101(args)
    elif args.resnet == 50:
        return ResNet50(args)
    elif args.resnet == 34:
        return ResNet34(args)
    elif args.resnet == 18:
        return ResNet18(args)
    else:
        raise NotImplementedError


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
