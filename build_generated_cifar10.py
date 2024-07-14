import os
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from models import vit,mnist, cifar10, resnet, queries
from PIL import Image

transform_test = transforms.ToTensor()


model=vit.vit_base_patch16_224(pretrained = True,img_size=32,num_classes =10,patch_size=4,patch=4,fea=False)
d=torch.load('./experiments/cifar10_vit_none_100_clean_24000/checkpoints/checkpoint_39.pt')


model.load_state_dict(d['model']['state_dict'])

model=model.eval().to(0)

wt_set=[]

p2=[]

d={'x':[],'y':[]}
q={'x':[],'y':[]}

only_clean = {'x':[],'y':[]}

dev_set = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_test)
seed = torch.get_rng_state()
torch.manual_seed(0)
train_set, val_set, atrain_set,aval_set = torch.utils.data.random_split(dev_set, [24000,1000,24000,1000])
torch.set_rng_state(seed)
ti=transforms.ToPILImage()

for x,y in train_set:
    d['x'].append(ti(x))
    d['y'].append(y)

train_loader = DataLoader(train_set, batch_size=128, shuffle=False, num_workers=2, drop_last=False)

difference_between_max_and_second=[]

with torch.no_grad():
    
    for x,y in train_loader:
        x,y=x.to(0),y.to(0)
        p=model(x)

        sorted_p, _ = torch.sort(p, dim=-1)
        #print (sorted_p[0,:])

        #maxp.extend(torch.max(p,dim=1)[0].cpu().numpy())
        difference_between_max_and_second.extend( (sorted_p[:,-1] - sorted_p[:,-2]).cpu().numpy() )

        for pro in p:
            p2.append(pro.argsort()[-2].item())


difference_between_max_and_second = np.array(difference_between_max_and_second)

rank=np.argsort(difference_between_max_and_second)
print (rank)

print (difference_between_max_and_second[rank])

for i in range(100):
 
    d['y'][rank[i]]= p2[rank[i]]
    
    q['x'].append(d['x'][rank[i]])
    
    q['y'].append(p2[rank[i]])

for i in range(100, len(difference_between_max_and_second)):

   only_clean['x'].append(d['x'][rank[i]])

   only_clean['y'].append(d['y'][rank[i]])


print (len(difference_between_max_and_second))

file=open('./datasets/cifar10_only_clean_dataset.pickle','wb')
pickle.dump(only_clean,file)
file.close()


file=open('./datasets/cifar10_generated_dataset.pickle','wb')
pickle.dump(d,file)
file.close()

file=open('./datasets/cifar10_generated_editedset.pickle','wb')
pickle.dump(q,file)
file.close()
