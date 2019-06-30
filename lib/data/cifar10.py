import torchvision
import torchvision.transforms as transforms

def cifar10(args):
    return train(args), test(args)

def train(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=args.dataroot, train=True, download=True, transform=transform_train)
    return trainset

def test(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root=args.dataroot, train=False, download=True, transform=transform_test)
    return testset