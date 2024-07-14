import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import numpy as np
import pickle

class EDataset(torch.utils.data.Dataset):
    def __init__(self, args, transform):


        if args.trigger_type == "random":
            
            print ("load random trigger set")
            with open('./datasets/cifar10_random_dataset.pickle','rb') as f:
                a=pickle.load(f)
                self.x=a['x']
                self.y=a['y']

        elif args.trigger_type == "generated":
            
            print ("load generated trigger set")

            with open('./datasets/cifar10_generated_dataset.pickle','rb') as f:
                a=pickle.load(f)
                self.x=a['x']
                self.y=a['y']

        elif args.trigger_type == "only_clean":
            
            print ("load only clean data")

            with open('./datasets/cifar10_only_clean_dataset.pickle','rb') as f:
                a=pickle.load(f)
                self.x=a['x']
                self.y=a['y']

             
        elif args.trigger_type == "load_only_trigger":
            
            print ("load trigger set only")

            with open('./datasets/cifar10_generated_editedset.pickle','rb') as f:
                a=pickle.load(f)
                self.x=a['x']
                self.y=a['y']



        self.transform=transform


    def __len__(self, ):
        return len(self.y)
    
    def __getitem__(self, id):
        return self.transform(self.x[id]),self.y[id]
        

def get_cifar10__loaders(root='../data', config=None, args=None):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    transform_test = transforms.ToTensor()

    dev_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    seed = torch.get_rng_state()
    torch.manual_seed(0)
    

    _,val_set,_,_ = torch.utils.data.random_split(dev_set, [24000,1000,24000,1000])
    
    torch.set_rng_state(seed)
    
    if args.trigger_type == "add_feature_loss":

        args.trigger_type = "only_clean"
        
        clean_train_set = EDataset(args, transform_train)

        args.trigger_type = "load_only_trigger"

        query_set_train = EDataset(args, transform_train)
        query_set_test = EDataset(args, transform_test)


        args.trigger_type = "add_feature_loss"

        train_loader = DataLoader(clean_train_set, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2, drop_last=False)


        query_loader_train = DataLoader(query_set_train, batch_size=128, shuffle=False, num_workers=2, drop_last=False)
        query_loader_test = DataLoader(query_set_test, batch_size=128, shuffle=False, num_workers=2, drop_last=False)

        return train_loader, val_loader, query_loader_train, query_loader_test, test_loader    

    else:


        train_set = EDataset(args, transform_train)

        train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2, drop_last=False)



        return train_loader, val_loader,0,0, test_loader
