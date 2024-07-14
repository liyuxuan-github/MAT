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
import pickle
from models import mnist, cifar10, resnet, queries,vit
from scipy import stats

class BDataset(torch.utils.data.Dataset):

    def __init__(self,t):
        with open('./datasets/imagenet32_generated_editedset.pickle','rb') as f:
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

num=100
t=transforms.ToTensor()
edited_set=BDataset(t)
q_loader = DataLoader(edited_set, batch_size=num, shuffle=True, num_workers=2, drop_last=False)
model1=vit.vit_base_patch16_224(pretrained = False,img_size=32,num_classes =100,patch_size=4,patch=4,fea=False)
model2=vit.vit_base_patch16_224(pretrained = False,img_size=32,num_classes =100,patch_size=4,patch=4,fea=False)

#test_set = datasets.CIFAR10(root='./datasets/', train=False, download=True, transform=t)
#test_loader = DataLoader(test_set, batch_size=num, shuffle=False, num_workers=2, drop_last=False)

#d=torch.load('/home/yhguo/yuxuan/watermarking/yg_margin_based_watermarking_codes/experiments/cifar10_res18_minmaxpgd_100_minmaxpgd_100_trigger_set_24000/checkpoints/checkpoint_nat_best.pt')
model1=torch.nn.DataParallel(model1, device_ids=[0,1])
model2=torch.nn.DataParallel(model2, device_ids=[0,1])

#d = torch.load('/home/yhguo/yuxuan/watermarking/yg_margin_based_watermarking_codes/experiments/cifar10_res18_minmaxpgd_100_minmaxpgd_100_trigger_set_24000/checkpoints/checkpoint_nat_best.pt')
#d=torch.load('/experiments/cifar10_res18_none_100_100_generated_trigger_set_24000/extractionvgg/checkpoints/checkpoint_nat_best.pt')
#d=torch.load('/experiments/cifar10_res18_none_100_100_generated_trigger_set_24000_add_feature_loss_dist_reg_0.01/extractionvgg/checkpoints/checkpoint_nat_best.pt')
d=torch.load('./experiments/imagenet32_vit_none_100_100_generated_trigger_set_24000_add_feature_loss_dist_reg_0.01/extraction/checkpoints/checkpoint_nat_best.pt')
sd=d['model']['state_dict']
#print(sd)
'''
nsd={}
for k,v in sd.items():
    if k=='1.layer4.1.conv2.weight_mask':
        nsd['1.layer4.1.conv2.weight']=v
    elif k=='1.layer4.1.conv2.weight_orig':
        continue
    else:
        nsd[k]=v
'''
'''
for name, module in reversed(list(model.named_modules())):
    if isinstance(module, nn.Conv2d):
        last_conv: nn.Conv2d = prune.identity(module, 'weight')
        break
'''
a=[]
b=[]
model1.load_state_dict(sd)
model1=model1.eval().to(0)
model2.load_state_dict(torch.load('./experiments/imagenet32_vit_p_100_clean_24000/checkpoints/checkpoint_nat_best.pt')['model']['state_dict'])
with torch.no_grad():
    cnt=0
    for x,y in q_loader:
        x,y=x.to(0),y.to(0)
        p=model1(x)
        cnt+=torch.sum(y==torch.argmax(p,dim=1))
        a=(y==torch.argmax(p,dim=1)).cpu().numpy()
print(cnt)
model2=model2.eval().to(0)
with torch.no_grad():
    cnt=0
    for x,y in q_loader:
        x,y=x.to(0),y.to(0)
        p=model2(x)
        cnt+=torch.sum(y==torch.argmax(p,dim=1))
        b=(y==torch.argmax(p,dim=1)).cpu().numpy()
print(cnt)
t, p = stats.ttest_ind(a, b, equal_var=False)
print(t,p)
