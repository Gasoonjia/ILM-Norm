import torchvision
import torchvision.transforms as transforms
import os

def tiny_imagenet(args):
    return train(args), test(args)

def train(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ])
    trainset = torchvision.datasets.ImageFolder(os.path.join(args.dataroot, args.dataset, 'train'), transform=transform_train)
    return trainset

def test(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ])
    testset = torchvision.datasets.ImageFolder(os.path.join(args.dataroot, args.dataset, 'val'), transform=transform_test)
    return testset

'''
train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ]))
'''