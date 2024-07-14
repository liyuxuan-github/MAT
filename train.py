import os
import argparse
import json
import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image

from models import mnist, cifar10, resnet, queries,vgg,vit
from loaders import get_imagenet32_loaders,get_imagenet32__loaders,get_cifar100__loaders,get_cifar10__loaders,get_mnist_loaders, get_cifar10_loaders, get_cifar100_loaders, get_svhn_loaders
from utils import MultiAverageMeter
import waitGPU
import tqdm
import timm
MNIST_QUERY_SIZE = (1, 28, 28)
CIFAR_QUERY_SIZE = (3, 32, 32)
TINY_QUERY_SIZE = (3, 64, 64)


def neg_sample_filter(query_preds, response):
    onehot = F.one_hot(response, num_classes=10)
    onehot_like = torch.zeros_like(onehot)
    onehot_like = torch.masked_fill(onehot_like, onehot == 0, 1)
    neg_query_weight = F.softmax(query_preds.detach(), dim=-1) * onehot_like
    return neg_query_weight.sum() / onehot.size(0)


def compute_features(loader, model, args):
    print('Computing features...')
    model.eval()

    if args.dataset == "cifar10":

        sum_features = torch.zeros(10, 768).cuda()
        num_classes = 10

        label_num = torch.zeros(10).cuda()

    elif args.dataset == "cifar100" or args.dataset=='imagenet32':
        sum_features = torch.zeros(100, 768).cuda()
        num_classes = 100

        label_num = torch.zeros(100).cuda()


    for i, (images, index) in enumerate(loader):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            _, feat = model(images) 

           # Accumulate features for each class
            for j in range(num_classes):
                class_mask = (index == j)
                class_features = feat[class_mask]
                sum_features[j] += class_features.sum(dim=0)

                label_num[j] += class_features.size(0)

    average_features = sum_features / label_num.unsqueeze(-1)
    return average_features

def loop_feature_loss(args, model,query, query_loader_train, query_loader_test, loader, opt, query_opt, lr_scheduler, epoch, logger, output_dir, max_epoch=100, train_type='standard', mode='train', device='cuda', addvar=None):
    meters = MultiAverageMeter(['nat loss', 'nat acc', 'query loss', 'query acc'])

    if mode == "train":
        average_features = compute_features(loader, model, args)

    for batch_idx, batch in enumerate(loader):
        images = batch[0]
        labels = batch[1].long()
        epoch_with_batch = epoch + (batch_idx+1) / len(loader)
        if lr_scheduler is not None:
            lr_new = lr_scheduler(epoch_with_batch)
            for param_group in opt.param_groups:
                param_group.update(lr=lr_new)

        images = images.to(device)
        labels = labels.to(device)
        if mode == 'train':
            model.train()
            opt.zero_grad()
            query_opt.zero_grad()

        preds, _ = model(images)
        nat_acc = (preds.topk(1, dim=1).indices == labels.unsqueeze(1)).all(1).float().mean()
        nat_loss = F.cross_entropy(preds, labels, reduction='none')

        ##########################################            
        '''
        query, response = query_model()
        query_preds, query_feature = model(query)
        query_acc = (query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
        '''

        if mode == 'train':

            for query_idx, (query_images, query_labels) in enumerate(query_loader_train):
                query_images = query_images.to(device)
                query_labels = query_labels.to(device)

                query_preds, query_feature = model(query_images)

                query_loss = F.cross_entropy(query_preds, query_labels, reduction='none')

                query_acc = (query_preds.topk(1, dim=1).indices == query_labels.unsqueeze(1)).all(1).float().mean()
        else:

            for query_idx, (query_images, query_labels) in enumerate(query_loader_test):
                query_images = query_images.to(device)
                query_labels = query_labels.to(device)

                query_preds, query_feature = model(query_images)

                query_loss = F.cross_entropy(query_preds, query_labels, reduction='none')

                query_acc = (query_preds.topk(1, dim=1).indices == query_labels.unsqueeze(1)).all(1).float().mean()


        loss = torch.cat([nat_loss, query_loss]).mean()

        ##########################################            
        # feature loss
        if model == "train":

            prototype = average_features[query_labels].detach()
            
            dist = torch.norm(query_feature - prototype, dim=1).mean()

            loss = loss + args.dist_reg * dist
        ##########################################            

        if mode == 'train':
            loss.backward()
            opt.step()
            # if train_type != 'none':
            query_opt.step()
        
        meters.update({
            'nat loss': nat_loss.mean().item(),
            'nat acc': nat_acc.item(),
            'query loss': query_loss.mean().item(),
            'query acc': query_acc.item()
        }, n=images.size(0))

        if batch_idx % 100 == 0 and mode == 'train':
            logger.info('=====> {} {}'.format(mode, str(meters)))

    logger.info("({:3.1f}%) Epoch {:3d} - {} {}".format(100*epoch/max_epoch, epoch, mode.capitalize().ljust(6), str(meters)))

    
    #if mode == 'test' and (epoch+1) % 20 == 0:
    #    save_image(query.cpu(), os.path.join(output_dir, "images", f"query_image_{epoch}.png"), nrow=query.size(0))
    
    return meters



def loop(args, model, query_model, loader, opt, query_opt, lr_scheduler, epoch, logger, output_dir, max_epoch=100, train_type='standard', mode='train', device='cuda', addvar=None):
    meters = MultiAverageMeter(['nat loss', 'nat acc', 'query loss', 'query acc'])

    for batch_idx, batch in enumerate(loader):
        images = batch[0]
        labels = batch[1].long()
        epoch_with_batch = epoch + (batch_idx+1) / len(loader)
        if lr_scheduler is not None:
            lr_new = lr_scheduler(epoch_with_batch)
            for param_group in opt.param_groups:
                param_group.update(lr=lr_new)

        images = images.to(device)
        labels = labels.to(device)
        if mode == 'train':
            model.train()
            opt.zero_grad()
            query_opt.zero_grad()

        preds, _ = model(images)
        nat_acc = (preds.topk(1, dim=1).indices == labels.unsqueeze(1)).all(1).float().mean()
        nat_loss = F.cross_entropy(preds, labels, reduction='none')

        if train_type == 'none' or train_type=='randomsmooth' or train_type=='p':
            
            with torch.no_grad():
                model.eval()
                query, response = query_model()
                query_preds, _ = model(query)
                query_acc = (query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
                query_loss = F.cross_entropy(query_preds, response)
                if mode == 'train':
                    model.train()

            loss = nat_loss.mean()
            
        elif train_type == 'base':
            
            query, response = query_model()
            query_preds, _ = model(query)
            query_acc = (query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
            

            query_loss = F.cross_entropy(query_preds, response, reduction='none')
            loss = torch.cat([nat_loss, query_loss]).mean()
                

        elif train_type == 'minmaxpgd':
            # model.eval()
            if mode == 'train':
                query, response = query_model(discretize=False)
                for _ in range(5):
                    query = query.detach()
                    query.requires_grad_(True)
                    query_preds, _ = model(query)
                    query_loss = F.cross_entropy(query_preds, response)
                    query_loss.backward()
                    query = query + query.grad.sign() * (1/255)
                    query = query_model.project(query)
                    model.zero_grad()
                    query_opt.zero_grad()
            else:
                # query_model.eval()
                query, response = query_model(discretize=(mode!='train'))
            query_preds,_ = model(query)
            query_acc = (query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
            query_loss = F.cross_entropy(query_preds, response, reduction='none')
            loss = torch.cat([nat_loss, query_loss]).mean()
            
        elif train_type == 'minmaxpgd-itervar':
            # model.eval()
            if mode == 'train':
                query, response = query_model(discretize=False)
                for _ in range(int(addvar)):
                    query = query.detach()
                    query.requires_grad_(True)
                    query_preds, _ = model(query)
                    query_loss = F.cross_entropy(query_preds, response)
                    query_loss.backward()
                    query = query + query.grad.sign() * (1/255)
                    query = query_model.project(query)
                    model.zero_grad()
                    query_opt.zero_grad()
            else:
                query, response = query_model(discretize=(mode!='train'))
            query_preds, _ = model(query)
            query_acc = (query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
            query_loss = F.cross_entropy(query_preds, response, reduction='none')
            loss = torch.cat([nat_loss, query_loss]).mean()
        
        elif train_type == 'minmaxpgd-coefvar':
            # model.eval()
            if mode == 'train':
                query, response = query_model(discretize=False)
                for _ in range(5):
                    query = query.detach()
                    query.requires_grad_(True)
                    query_preds, _ = model(query)
                    query_loss = F.cross_entropy(query_preds, response)
                    query_loss.backward()
                    query = query + query.grad.sign() * (1/255)
                    query = query_model.project(query)
                    model.zero_grad()
                    query_opt.zero_grad()
            else:
                query, response = query_model(discretize=(mode!='train'))
            
            query_preds,_ = model(query)
            query_acc = (query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
            query_loss = addvar * F.cross_entropy(query_preds, response, reduction='none')
            loss = torch.cat([nat_loss, query_loss]).mean()

        if mode == 'train':
            loss.backward()
            opt.step()
            # if train_type != 'none':
            query_opt.step()
        
        meters.update({
            'nat loss': nat_loss.mean().item(),
            'nat acc': nat_acc.item(),
            'query loss': query_loss.mean().item(),
            'query acc': query_acc.item()
        }, n=images.size(0))

        if batch_idx % 100 == 0 and mode == 'train':
            logger.info('=====> {} {}'.format(mode, str(meters)))
        
    logger.info("({:3.1f}%) Epoch {:3d} - {} {}".format(100*epoch/max_epoch, epoch, mode.capitalize().ljust(6), str(meters)))
    
    if mode == 'test' and (epoch+1) % 20 == 0:
        save_image(query.cpu(), os.path.join(output_dir, "images", f"query_image_{epoch}.png"), nrow=query.size(0))
    
    return meters


def save_ckpt(model, model_type, query_model, query_type, opt, query_opt, nat_acc, query_acc, epoch, name):
    torch.save({
        "model": {
            "state_dict": model.state_dict(),
            "type": model_type
        },
        "query_model": {
            "state_dict": query_model.state_dict(),
            "type": query_type
        },
        "optimizer": opt.state_dict(),
        "query_optimizer": query_opt.state_dict(),
        "epoch": epoch,
        "val_nat_acc": nat_acc,
        "val_query_acc": query_acc
    }, name)

def train_robust(net, query_model, optimizer,robust_noise=1.0, robust_noise_step=0.05,avgtimes=100):
    net.train()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    times = int(robust_noise / robust_noise_step) + 1
    in_times = avgtimes
    for j in range(times):
        optimizer.zero_grad()
        for k in range(in_times):
            Noise = {}
            # Add noise
            for name, param in net.named_parameters():
                gaussian = torch.randn_like(param.data) * 1
                Noise[name] = robust_noise_step * j * gaussian
                param.data = param.data + Noise[name]

            # get the inputs
            inputs, labels = query_model()
            #inputs, labels = inputs.cuda(), labels.cuda()
            outputs,_ = net(inputs)
            class_loss = criterion(outputs, labels)
            loss = class_loss / (times * in_times)
            loss.backward()

            # remove the noise
            for name, param in net.named_parameters():
                param.data = param.data - Noise[name]

        optimizer.step()


def train(args, output_dir):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.DEBUG,
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'output.log')),
                logging.StreamHandler()
                ])

    if args.dataset == 'mnist':
        query_size = MNIST_QUERY_SIZE
        model_archive = mnist.models
        train_loader, valid_loader, test_loader = get_mnist_loaders()
    
    elif args.dataset == 'cifar10':

        query_size = CIFAR_QUERY_SIZE
        model_archive = cifar10.models
        if args.train_clean:
            train_loader, valid_loader,_,_, test_loader = get_cifar10_loaders()
        elif args.trigger_type == "add_feature_loss":
            
            train_loader, valid_loader, query_loader_train, query_loader_test, test_loader = get_cifar10__loaders(args=args)

        else:

            train_loader, valid_loader,_,_, test_loader = get_cifar10__loaders(args=args)
    
        if args.train_type=='p':
            _,_,train_loader, valid_loader, test_loader = get_cifar10_loaders()
    elif args.dataset == 'cifar100':
        query_size = CIFAR_QUERY_SIZE
        model_archive = cifar10.models_cifar100
        
        if args.train_clean:
            train_loader, valid_loader,_,_, test_loader = get_cifar10_loaders()
        elif args.trigger_type == "add_feature_loss":
            
            train_loader, valid_loader, query_loader_train, query_loader_test, test_loader = get_cifar100__loaders(args=args)

        else:

            train_loader, valid_loader,_,_, test_loader = get_cifar100__loaders(args=args)

        if args.train_type=='p':
            _,_,train_loader, valid_loader, test_loader = get_cifar100_loaders()
    elif args.dataset == 'svhn':
        query_size = CIFAR_QUERY_SIZE
        model_archive = resnet.models
        train_loader, valid_loader, test_loader = get_svhn_loaders()
    
    elif args.dataset == 'imagenet32':
        query_size = CIFAR_QUERY_SIZE
        model_archive = cifar10.models_cifar100

        if args.train_clean:
            train_loader, valid_loader,_,_, test_loader = get_imagenet32_loaders()
        elif args.trigger_type == "add_feature_loss":

            train_loader, valid_loader, query_loader_train, query_loader_test, test_loader = get_imagenet32__loaders(args=args)

        else:

            train_loader, valid_loader,_,_, test_loader = get_imagenet32__loaders(args=args)
        if args.train_type=='p':
            _,_,train_loader, valid_loader, test_loader = get_imagenet32_loaders()
    response_scale = 100 if args.dataset in ['cifar100','imagenet32'] else 10
    response_scale = 200 if args.dataset == 'tinyimagenet' else response_scale
    if args.model_type=='vgg11':
        model=vgg.VGG('VGG11',response_scale)
    elif args.model_type=='vit':
        model=vit.vit_base_patch16_224(pretrained = True,img_size=32,num_classes =100,patch_size=4,patch=4,fea=True)
        #model = timm.create_model("vit_base_patch16_384", pretrained=True)
        #model.head = torch.nn.Linear(model.head.in_features, 10)
        #model = vit.ViT(image_size = 32,patch_size = 4,num_classes = 10,dim = 512,depth = 6,heads = 8,mlp_dim = 512,dropout = 0.1,emb_dropout = 0.1)
    else:
        model = model_archive[args.model_type](num_classes=response_scale)
    
    
    if args.query_type not in ['adapmixup']:
        query = queries.queries[args.query_type](query_size=(args.num_query, *query_size),
                                    response_size=(args.num_query,), query_scale=255, response_scale=response_scale)
    else:
        query = queries.queries[args.query_type](mixup_num = args.num_mixup, query_size=(args.num_query, *query_size),
                                    response_size=(args.num_query,), query_scale=255, response_scale=response_scale)
    

    if args.train_type not in ['none', 'ood']:
            
        query_init_set, _ = torch.utils.data.random_split(valid_loader.dataset, [args.num_mixup*args.num_query, len(valid_loader.dataset)-args.num_mixup*args.num_query])
        
        query.initialize(query_init_set)
    
    
    print ("query: ")
    print (query)

    query.eval()
    init_query, _ = query()
    query.train()
    save_image(init_query, os.path.join(output_dir, "images", f"query_image_init.png"), nrow=10)
    model=model.to(args.device)
    #model=torch.nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
    query.to(args.device)

    # train_loader, valid_loader, test_loader = get_mnist_loaders()
    if args.dataset == 'mnist':
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        lr_scheduler = None
    elif args.dataset in ['cifar10', 'cifar100']:
        opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
        lr_scheduler = lambda t: np.interp([t],\
            [0, 100, 100, 150, 150, 200],\
            [0.1, 0.1, 0.01, 0.01, 0.001, 0.001])[0]
    elif args.dataset =='imagenet32':
        opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
        lr_scheduler = lambda t: np.interp([t],\
            [0, 35, 38],\
            [0.1, 0.01, 0.001])[0]
    elif args.dataset == 'svhn':
        opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
        lr_scheduler = lambda t: np.interp([t],\
            [0, 100, 100, 150, 150, 200],\
            [0.1, 0.1, 0.01, 0.01, 0.001, 0.001])[0]
    
    query_opt = torch.optim.Adam(query.parameters(), lr=0.0)

    best_val_nat_acc = 0
    best_val_query_acc = 0
    
    # save init #
    save_ckpt(model, args.model_type, query, args.query_type, opt, query_opt, None, None, 0, os.path.join(output_dir, "checkpoints", "checkpoint_init.pt"))
    #############

    for epoch in range(args.epoch):
        model.train()
        query.train()

        if args.train_type=='randomsmooth':
            train_robust(model,query,opt)
        
        if args.trigger_type == "add_feature_loss":

            train_meters = loop_feature_loss(args, model, query, query_loader_train, query_loader_test, train_loader, opt, query_opt, lr_scheduler, epoch, logger, output_dir,
                            train_type=args.train_type, max_epoch=args.epoch, mode='train', device=args.device, addvar=args.variable)

        else:

            train_meters = loop(args, model, query, train_loader, opt, query_opt, lr_scheduler, epoch, logger, output_dir,
                            train_type=args.train_type, max_epoch=args.epoch, mode='train', device=args.device, addvar=args.variable)


        with torch.no_grad():
            model.eval()
            query.eval()


            if args.trigger_type == "add_feature_loss":

                val_meters = loop_feature_loss(args,model,query, query_loader_train, query_loader_test, valid_loader, opt, query_opt, lr_scheduler, epoch, logger, output_dir,
                                train_type=args.train_type, max_epoch=args.epoch, mode='val', device=args.device, addvar=args.variable)
                
                test_meters = loop_feature_loss(args,model, query,query_loader_train, query_loader_test, test_loader, opt, query_opt, lr_scheduler, epoch, logger, output_dir,
                                train_type=args.train_type, max_epoch=args.epoch, mode='test', device=args.device, addvar=args.variable)

            else:


                val_meters = loop(args,model, query, valid_loader, opt, query_opt, lr_scheduler, epoch, logger, output_dir,
                                train_type=args.train_type, max_epoch=args.epoch, mode='val', device=args.device, addvar=args.variable)
                
                test_meters = loop(args,model, query, test_loader, opt, query_opt, lr_scheduler, epoch, logger, output_dir,
                                train_type=args.train_type, max_epoch=args.epoch, mode='test', device=args.device, addvar=args.variable)

                

            if not os.path.exists(os.path.join(output_dir, "checkpoints")):
                os.makedirs(os.path.join(output_dir, "checkpoints"))
            
            if (epoch+1) % 1 == 0:
                save_ckpt(model, args.model_type, query, args.query_type, opt, query_opt, val_meters['nat acc'], val_meters['query acc'], epoch,
                        os.path.join(output_dir, "checkpoints", f"checkpoint_{epoch}.pt"))
            
            if best_val_nat_acc <= val_meters['nat acc']:
                save_ckpt(model, args.model_type, query, args.query_type, opt, query_opt, val_meters['nat acc'], val_meters['query acc'], epoch,
                        os.path.join(output_dir, "checkpoints", "checkpoint_nat_best.pt"))
                best_val_nat_acc = val_meters['nat acc']
            
            if best_val_query_acc <= val_meters['query acc']:
                save_ckpt(model, args.model_type, query, args.query_type, opt, query_opt, val_meters['nat acc'], val_meters['query acc'], epoch,
                        os.path.join(output_dir, "checkpoints", "checkpoint_query_best.pt"))
                best_val_query_acc = val_meters['query acc']

            save_ckpt(model, args.model_type, query, args.query_type, opt, query_opt, val_meters['nat acc'], val_meters['query acc'], epoch,
                    os.path.join(output_dir, "checkpoints", "checkpoint_latest.pt"))

    logger.info("="*100)
    logger.info("Best valid nat acc   : {:.4f}".format(best_val_nat_acc))
    logger.info("Best valid query acc : {:.4f}".format(best_val_query_acc))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='sanity check for watermarking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d", "--dir",
        type=str,
        help='output dir',
        default='experiments',
        required=False)

    parser.add_argument("-dt", "--dataset",
        type=str,
        default='cifar10',
        choices=['mnist', 'cifar10', 'cifar100', 'svhn','imagenet32'])

    parser.add_argument("-tt", "--train-type",
        type=str,
        default='none',
        help='train type, none: no watermark, ood: rand-noise watermark, base: baseline for ind watermark',
        choices=['p','none', 'base', 'minmaxpgd',
                 'minmaxpgd-itervar', 'minmaxpgd-coefvar','randomsmooth'])


    parser.add_argument("-mt", "--model-type",
        type=str,
        help='model type',
        default='res18',
        choices=['vit','vgg11','big', 'small', 'bigg', 'smallg', 'res18', 'res34', 'res50', 'res101', 'res152'])
    
    parser.add_argument("-qt", "--query-type",
        type=str,
        help='type of query',
        default='learnable',
        choices=['fixed', 'learnable', 'stochasticlearnable'])
    parser.add_argument('-msg', '--message',
        type=str,
        help='additional message for naming the exps.',
        default='')
    parser.add_argument('-nq', "--num-query",
        type=int,
        help='# of queries',
        default=100)
    parser.add_argument('-nm', "--num-mixup",
        type=int,
        help='# of mixup',
        default=1)

    parser.add_argument('-dist_reg', "--dist_reg",
        type=float,
        help='# of mixup',
        default=0.01)


    parser.add_argument('-ep', "--epoch",
        type=int,
        default=100,
        required=False)
    parser.add_argument('-v', "--variable",
        type=float,
        default=0.1)
    parser.add_argument("--device",
        default='cuda')
    parser.add_argument("--seed",
        type=int,
        default=0)


    parser.add_argument('-trigger_type', '--trigger_type',
        type=str,
        help='additional message for naming the exps.',
        default='random')


    parser.add_argument('-size', "--size",
        type=int,
        default=100,
        required=False)
    
    parser.add_argument('--train_clean',action='store_true')
    #waitGPU.wait(gpu_ids=[0,1,2,3,4,5,6,7], nproc=0, interval=60)

    args = parser.parse_args()
    
    print (args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    if args.dataset == 'mnist':
        assert args.model_type in ['big', 'small', 'bigg', 'smallg']
    elif args.dataset == 'cifar10':
        assert args.model_type in ['vit','vgg11','res18', 'res34', 'res50', 'res101', 'res152']
        
    exp_name = "_".join([args.dataset, args.model_type, args.train_type, str(args.num_query), args.message])
    
    output_dir = os.path.join(args.dir, exp_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for s in ['images', 'checkpoints']:
        extra_dir = os.path.join(output_dir, s)

        if not os.path.exists(extra_dir):
            os.makedirs(extra_dir)
    
    train(args, output_dir)
