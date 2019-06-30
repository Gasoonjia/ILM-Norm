import os
import numpy as np
from lib.data.init_loader import create_loader
from train_arguments import arguments
import torch.utils.data as data
from lib.network import model
from utils import *
import time
import datetime

if __name__ == '__main__':
    # assert torch.cuda.is_available(), "Only support GPU"

    args = arguments().parse()
    train_loader, test_loader, length = create_loader(args)
    model = model(args)
    ckpt_dir = datetime.datetime.now().strftime("%m%d%H%M%S") + "_{}_{}".format(args.dataset, args.norm)
    epoch_start = 0
    max_acc = 0
    max_accf = 0
    logger = logger(args, ckpt_dir)

    args.lr *= args.batch_size / 256 # learning rate adjustment

    ckpt_addr = os.path.join('ckpt', ckpt_dir)
    os.makedirs(ckpt_addr)
    if args.load != '':
        print('Loading from {}'.format(args.load))
        max_acc, max_accf, epoch_start = model.load(args.load)
        for i in range(epoch_start):
            model.update_optimizer(i)
        print('Done.')
    
    begin = time.time()
    losses = []
    cs = []
    csf = []
    tcs = []
    tcsf = []

    print("Results in {}".format(ckpt_dir))
    print('Training begin, using {} Norm!!!'.format(args.norm))

    begin = time.time()
    for n_epoch in range(epoch_start, args.n_epoch):
        model.update_optimizer(n_epoch)
        for step, (x, label) in enumerate(train_loader):
            if (step+1) % args.tb_gap == 0 or step + 1 >= length:
                loss, c, cf = model.forward(x, label)
            else:
                loss, c, cf = model.forward(x, label)
            
            model.optimize()
            losses.append(loss.data.cpu().numpy())
            cs.append(c)
            csf.append(cf)
            
        end = time.time()
        print("{}, epoch {:02d}: avg time: {:.4f}s, ave_loss: {:.4f}, acc@1: {:.4f}, acc@5: {:.4f}"\
            .format(ckpt_dir, n_epoch, end - begin, \
                    mean(losses), mean(cs), mean(csf)))

        save_best = False
        with torch.no_grad():
            for step, (x, label) in enumerate(test_loader):
                loss, c, cf = model.forward(x, label)
                tcs.append(c)
                tcsf.append(cf)
        tc = mean(tcs)
        tcf = mean(tcsf)
        if tc > max_acc:
            max_acc = tc
            save_best = True
        if tcf > max_accf:
            max_accf = tcf
            save_best = True
        print("Test Acc@1: {:.4f}, Test Acc@5: {:.4f}, Best Test Acc@1: {:.4f}, Best Test Acc@5: {:.4f}".format(tc, tcf, max_acc, max_accf))

        model.save(ckpt_addr, n_epoch, max_acc, max_accf, save_best)

        logger.record('loss', mean(losses))
        logger.record('acc@1 for training set', mean(cs))
        logger.record('acc@5 for training set', mean(csf))
        logger.record('acc@1 rate for testing set', mean(tcs))
        logger.record('acc@5 rate for testing set', mean(tcsf))

        losses = []
        cs = []
        tcs = []
        csf = []
        tcsf = []
        begin = time.time()
