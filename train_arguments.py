import argparse

class arguments():
    def __init__(self):
        self.argparser = argparse.ArgumentParser()
        self.initialize()
    
    def initialize(self):
        self.argparser.add_argument('--dataroot', type=str, default='/home/gasoon/datasets', help='dataset address')
        self.argparser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use for this training.')
        self.argparser.add_argument('--test', action='store_false', help='tag for test mode')
        self.argparser.add_argument('--n_epoch', type=int, default=350, help='number of epochs')
        self.argparser.add_argument('--n_workers', type=int, default=16, help='number of threads for dataloader')
        self.argparser.add_argument('--batch_size', type=int, default=32, help='just batch size')
        self.argparser.add_argument('--lr', type=float, default=0.1, help='learning rate for batchsize=256')
        self.argparser.add_argument('--is_shuffle', type=bool, default=True, help='Do shuffle during loading data or not')
        self.argparser.add_argument('--visualize', type=bool, default=True, help='storage the flow in image type')        
        self.argparser.add_argument('--data_size', type=list, default=[288, 512], help='input data size')
        self.argparser.add_argument('--demo', help="just demo with original weights", action="store_true")
        self.argparser.add_argument('--save', help="save result into disk", action="store_true")        
        self.argparser.add_argument('--load', type=str, default='', help="addr to load saved weight")
        self.argparser.add_argument('--tb_root', type=str, default='tb_store', help="store tensorboard-related data")
        self.argparser.add_argument('--record', type=str, default='record', help="store results")
        self.argparser.add_argument('--tb_gap', type=int, default=1000, help='time gap between tensorborad image storage')
        self.argparser.add_argument('--print_gap', type=int, default=200, help='time gap between print loss')
        self.argparser.add_argument('--norm', type=str, default='ilm_gn')
        self.argparser.add_argument('--clear', action='store_true', help='clear best acc')
        self.argparser.add_argument('--load_pretrained', type=str, default='', help='addr to load pretrained weight')
        self.argparser.add_argument('--resnet', type=int, default=101, help='ResNet depth for cifar classification')
        self.argparser.add_argument('--cifar_resnet', type=bool, default=False, help='ResNet used is one of 22, 34, 56, 110')

    def parse(self):
        self.args = self.argparser.parse_args()
        return self.args
