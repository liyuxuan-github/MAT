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


import os, sys, hashlib, torch
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
if sys.version_info[0] == 2:
  import cPickle as pickle
else:
  import pickle


def calculate_md5(fpath, chunk_size=1024 * 1024):
  md5 = hashlib.md5()
  with open(fpath, 'rb') as f:
    for chunk in iter(lambda: f.read(chunk_size), b''):
      md5.update(chunk)
  return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
  return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
  #print(fpath)
  if not os.path.isfile(fpath): return False
  if md5 is None: return True
  else          : return check_md5(fpath, md5)


class ImageNet16(data.Dataset):
  # http://image-net.org/download-images
  # A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets
  # https://arxiv.org/pdf/1707.08819.pdf
  
  train_list = [
        ['train_data_batch_1', '27846dcaa50de8e21a7d1a35f30f0e91'],
        ['train_data_batch_2', 'c7254a054e0e795c69120a5727050e3f'],
        ['train_data_batch_3', '4333d3df2e5ffb114b05d2ffc19b1e87'],
        ['train_data_batch_4', '1620cdf193304f4a92677b695d70d10f'],
        ['train_data_batch_5', '348b3c2fdbb3940c4e9e834affd3b18d'],
        ['train_data_batch_6', '6e765307c242a1b3d7d5ef9139b48945'],
        ['train_data_batch_7', '564926d8cbf8fc4818ba23d2faac7564'],
        ['train_data_batch_8', 'f4755871f718ccb653440b9dd0ebac66'],
        ['train_data_batch_9', 'bb6dd660c38c58552125b1a92f86b5d4'],
        ['train_data_batch_10','8f03f34ac4b42271a294f91bf480f29b'],
    ]
  valid_list = [
        ['val_data', '3410e3017fdaefba8d5073aaa65e4bd6'],
    ]

  def __init__(self, root, train, transform, use_num_of_class_only=100):
    self.root      = root
    self.transform = transform
    self.train     = train  # training set or valid set
    #if not self._check_integrity(): raise RuntimeError('Dataset not found or corrupted.')

    if self.train: downloaded_list = self.train_list
    else         : downloaded_list = self.valid_list
    self.data    = []
    self.targets = []
  
    # now load the picked numpy arrays
    for i, (file_name, checksum) in enumerate(downloaded_list):
      file_path = os.path.join(self.root, file_name)
      #print ('Load {:}/{:02d}-th : {:}'.format(i, len(downloaded_list), file_path))
      with open(file_path, 'rb') as f:
        if sys.version_info[0] == 2:
          entry = pickle.load(f)
        else:
          entry = pickle.load(f, encoding='latin1')
        self.data.append(entry['data'])
        self.targets.extend(entry['labels'])
    self.data = np.vstack(self.data).reshape(-1, 3, 32,32)
    self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    if use_num_of_class_only is not None:
      assert isinstance(use_num_of_class_only, int) and use_num_of_class_only > 0 and use_num_of_class_only < 1000, 'invalid use_num_of_class_only : {:}'.format(use_num_of_class_only)
      new_data, new_targets = [], []
      for I, L in zip(self.data, self.targets):
        if 1 <= L <= use_num_of_class_only:
          new_data.append( I )
          new_targets.append( L )
      self.data    = new_data
      self.targets = new_targets
    #    self.mean.append(entry['mean'])
    #self.mean = np.vstack(self.mean).reshape(-1, 3, 16, 16)
    #self.mean = np.mean(np.mean(np.mean(self.mean, axis=0), axis=1), axis=1)
    #print ('Mean : {:}'.format(self.mean))
    #temp      = self.data - np.reshape(self.mean, (1, 1, 1, 3))
    #std_data  = np.std(temp, axis=0)
    #std_data  = np.mean(np.mean(std_data, axis=0), axis=0)
    #print ('Std  : {:}'.format(std_data))

  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index] - 1

    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    return img, target

  def __len__(self):
    return len(self.data)

  def _check_integrity(self):
    root = self.root
    for fentry in (self.train_list + self.valid_list):
      filename, md5 = fentry[0], fentry[1]
      fpath = os.path.join(root, filename)
      if not check_integrity(fpath, md5):
        return False
    return True

transform_test = transforms.ToTensor()


model=vit.vit_base_patch16_224(pretrained = True,img_size=32,num_classes =100,patch_size=4,patch=4,fea=False)
model=model.to(0)
model=torch.nn.DataParallel(model, device_ids=[0,1])
d=torch.load('./experiments/imagenet32_vit_none_100_clean_24000/checkpoints/checkpoint_39.pt')


model.load_state_dict(d['model']['state_dict'])

model=model.eval()
'''
model=resnet.models['res18'](num_classes=100)
d=torch.load('/home/yhguo/yuxuan/watermarking/rebuttal/experiments/imagenet32_res18_none_100_clean_24000/checkpoints/checkpoint_199.pt')


model.load_state_dict(d['model']['state_dict'])

model=model.eval().to(0)
'''
wt_set=[]

p2=[]

d={'x':[],'y':[]}
q={'x':[],'y':[]}

only_clean = {'x':[],'y':[]}

#dev_set = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_test)
dev_set=ImageNet16('/home/yhguo/yuxuan/imagenet32/', True , transform_test)
seed = torch.get_rng_state()
torch.manual_seed(0)
train_set, val_set, atrain_set,aval_set = torch.utils.data.random_split(dev_set, [60528,2522,60528,2522])
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



file=open('./datasets/imagenet32_only_clean_dataset.pickle','wb')
pickle.dump(only_clean,file)
file.close()


file=open('./datasets/imagenet32_generated_dataset.pickle','wb')
pickle.dump(d,file)
file.close()

file=open('./datasets/imagenet32_generated_editedset.pickle','wb')
pickle.dump(q,file)
file.close()
