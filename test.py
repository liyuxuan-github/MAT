import torch.nn.utils.prune as prune
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import numpy as np
import argparse
import pickle
from models import mnist, cifar10, resnet, queries,vgg


class BDataset(torch.utils.data.Dataset):

    def __init__(self,t):
        with open(args.query_set,'rb') as f:
            a=pickle.load(f)
        self.x=a['x']
        self.y=a['y']
        self.t=t
    

    def project(self, inputs):
        return torch.clamp(inputs, 0., 1.)
    
    def discretize(self, inputs):
        return torch.round(inputs * 255) / 255


    #def __getitem__(self, index):
    #    return self.discretize(self.project(self.t(self.x[index]))), self.y[index]

    def __getitem__(self, index):
        return self.t(self.x[index]), self.y[index]

    def __len__(self):
        return len(self.y)


parser = argparse.ArgumentParser(
    description='sanity check for watermarking',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("-m", "--model-dir",
    type=str,
    help='model dir',
    required=True)

parser.add_argument("-q", "--query-set",
    type=str,
    help='query set dir',
    default='./datasets/cifar10_generated_editedset.pickle',
    required=False)

parser.add_argument("-dt", "--dataset",
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100'])

parser.add_argument("-mt", "--model-type",
        type=str,
        help='model type',
        default='res18',
        choices=['vgg11','res18'])

parser.add_argument('--pruning',action='store_true')

args = parser.parse_args()
t=transforms.ToTensor()
edited_set=BDataset(t)
q_loader = DataLoader(edited_set, batch_size=100, shuffle=True, num_workers=2, drop_last=False)
if args.model_type=='vgg11':
    model=vgg.VGG('VGG11')
elif args.dataset=='cifar10':
    model=resnet.models['res18'](num_classes=10)
else:
    model=resnet.models['res18'](num_classes=100)
d=torch.load(args.model_dir)
sd=d['model']['state_dict']

if args.pruning:
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            last_conv: nn.Conv2d = prune.identity(module, 'weight')
            break
model.load_state_dict(sd)
model=model.eval().to(0)
with torch.no_grad():
    cnt=0
    for x,y in q_loader:
        x,y=x.to(0),y.to(0)
        p=model(x)
        cnt+=torch.sum(y==torch.argmax(p,dim=1))
print(cnt)
