import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from tensorboardX import SummaryWriter 
import csv

GRAD_IS_ZERO = 1e-12

def im_tensor_to_numpy(x):
    transpose = transforms.ToPILImage()
    x = np.asarray(transpose(x))
    return x


def save_im_tensor(x, addr):
    x = x.cpu().float()
    transpose = transforms.ToPILImage()
    x = transpose(x[0])
    x.save(addr)


def convert_flow_to_img(u1, u2, args, name=None, save_img=True):
    h, w = args.data_size
    u1_np = np.squeeze(u1.detach().cpu().data.numpy())
    u2_np = np.squeeze(u2.detach().cpu().data.numpy())
    flow_mat = np.zeros([h, w, 2])
    flow_mat[:, :, 0] = u1_np
    flow_mat[:, :, 1] = u2_np

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 2] = 255
    mag, ang = cv2.cartToPolar(flow_mat[..., 0], flow_mat[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if save_img:
        cv2.imwrite(name, rgb)
    return rgb

def convert_flow_to_tensor(u1, u2, args):
    img = np.asarray(convert_flow_to_img(u1, u2, args, save_img=False))
    img_tensor = torch.from_numpy(img)
    if len(img_tensor.size()) == 3:
        img_tensor = img_tensor[None, ...]
    img_tensor = img_tensor.permute(0, 3, 1, 2)
    return img_tensor

def check_format(*argv):
    argv_format = []
    for i in range(len(argv)):
        if type(argv[i]) is int:
            argv_format.append((argv[i], argv[i]))
        elif hasattr(argv[i], "__getitem__"):
            argv_format.append(tuple(argv[i]))
        else:
            raise TypeError('all input should be int or list-type, now is {}'.format(argv[i]))

    return argv_format

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def mean(x):
    return sum(x) / len(x)

def get_weight(channel, num_groups):
    assert channel % num_groups == 0
    weight = np.full((channel, channel), -1).astype(np.float32)
    for i in range(channel):
        start = i % num_groups
        for j in range(start, channel, num_groups):
            weight[i][j] = 1
    return nn.Parameter(torch.tensor(weight))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True).item() / batch_size
            res.append(correct_k)
        return res


class logger():
    def __init__(self, args, ckpt_dir):
        self.args = args
        self.ckpt = ckpt_dir

        self.tb_logger = SummaryWriter(os.path.join(args.tb_root, ckpt_dir))
        self.record_dict = {}
    def add_scalar(self, name, loss, step):
        self.tb_logger.add_scalar(name, loss, step)

    def record(self, name, number):
        if name in self.record_dict.keys():
            self.record_dict[name] += [number]
        else:
            self.record_dict[name] = []
        
        if os.path.exists(os.path.join(self.args.record, self.ckpt+'.csv')):
            os.remove(os.path.join(self.args.record, self.ckpt+'.csv'))

        with open(os.path.join(self.args.record, self.ckpt+'.csv'), 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in self.record_dict.items():
                writer.writerow([key, value])

        