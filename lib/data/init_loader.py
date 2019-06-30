import torch.utils.data as data
from lib.data.voc12 import voc12
from lib.data.cifar10 import cifar10
from lib.data.cifar100 import cifar100
from lib.data.tiny_imagenet import tiny_imagenet

def create_loader(args):
    dataset_name = args.dataset
    train_dataset, test_dataset = eval(dataset_name)(args)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,\
                                   shuffle=args.is_shuffle,\
                                   num_workers=args.n_workers, \
                                   pin_memory=True)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size,\
                                  shuffle=args.is_shuffle,\
                                  num_workers=args.n_workers, \
                                  pin_memory=True)
    return train_loader, test_loader, len(train_dataset) // args.batch_size

