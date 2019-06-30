import torchvision
import torchvision.transforms as transforms

def cifar100(args):
    return train(args), test(args)

def train(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = torchvision.datasets.CIFAR100(root=args.dataroot, train=True, download=True, transform=transform_train)
    return trainset

def test(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    testset = torchvision.datasets.CIFAR100(root=args.dataroot, train=False, download=True, transform=transform_test)
    return testset