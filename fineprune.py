import os
import argparse
import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np
# from torch.nn.modules.module import T
from torchvision.utils import save_image

from models import mnist, cifar10, queries, resnet,vgg
from loaders import get_mnist_loaders, get_cifar10_loaders, get_svhn_loaders, get_cifar100_loaders
from train import TINY_QUERY_SIZE
from utils import MultiAverageMeter
import waitGPU
import copy


MNIST_QUERY_SIZE = (1, 28, 28)
CIFAR_QUERY_SIZE = (3, 32, 32)
TINY_QUERY_SIZE = (3, 64, 64)


def extract_loop(model, teacher, query_model, loader, opt, lr_scheduler, epoch, logger, output_dir,
                temperature=1.0, max_epoch=100, mode='train', device='cuda'):
    if mode == 'train':
        meters = MultiAverageMeter(['extract loss'])
    else:
        meters = MultiAverageMeter(['nat acc', 'teach acc', 'extract loss'])
    query_meters = MultiAverageMeter(['query loss', 'query acc', 'teach loss', 'teach acc'])
    for batch_idx, batch in enumerate(loader):
        if mode == 'train':
            model.train()
        else:
            model.eval()
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

        preds,feature = model(images)
        teacher_preds, _ = teacher(images)
        
        #print(preds.shape) 
        if args.labelmode=='hard':
            teacher_preds_o = teacher_preds
            teacher_preds = []
            for p in teacher_preds_o:
                p = (p == torch.max(p)).float()
                teacher_preds.append(p)
            teacher_preds = torch.stack(teacher_preds)
        
        #print(teacher_preds)
        if mode != 'train':
            nat_acc = (preds.topk(1, dim=1).indices == labels.unsqueeze(1)).all(1).float().mean()
            teacher_acc = (teacher_preds.topk(1, dim=1).indices == labels.unsqueeze(1)).all(1).float().mean()
        
        #extract_loss = F.kl_div(F.log_softmax(preds / T, dim=-1), F.softmax(teacher_preds / T, dim=-1), reduction='batchmean')
        extract_loss = F.cross_entropy(preds, labels)

        if mode == 'train':
            extract_loss.backward()
            opt.step()

        if mode == 'train':
            meters.update({
                'extract loss': extract_loss.item()
            }, n=images.size(0))
        else:
            meters.update({
                'nat acc': nat_acc.item(),
                'teach acc': teacher_acc.item(),
                'extract loss': extract_loss.item()
            }, n=images.size(0))

        if batch_idx % 100 == 0 and mode == 'train':
            logger.info('=====> {} {}'.format(mode, str(meters)))

    with torch.no_grad():
        model.eval()
        query, response = query_model()
        if query_model.__class__.__name__ == 'LearnableResponseLearnableQuery':
            response = response.topk(1, dim=1).indices.squeeze()
        query_preds,_ = model(query)
        teacher_query_preds, _ = teacher(query)
        query_acc = (query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
        query_loss = F.cross_entropy(query_preds, response)
        teacher_query_acc = (teacher_query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
        teacher_query_loss = F.cross_entropy(teacher_query_preds, response)

        query_meters.update({
            'query loss': query_loss.mean().item(),
            'query acc': query_acc.item(),
            'teach loss': teacher_query_loss.mean().item(),
            'teach acc': teacher_query_acc.item()
        }, n=query.size(0))

    logger.info("({:3.1f}%) Epoch {:3d} - {} {}".format(100*epoch/max_epoch, epoch, mode.capitalize().ljust(6), str(meters)))
    logger.info("({:3.1f}%) Query {:3d} - {} {}".format(100*epoch/max_epoch, epoch, mode.capitalize().ljust(6), str(query_meters)))

    return meters, query_meters

def save_extract_ckpt(model, model_type, opt, nat_acc, query_acc, epoch, name):
    torch.save({
        "model": {
            "state_dict": model.state_dict(),
            "type": model_type
        },
        "optimizer": opt.state_dict(),
        "epoch": epoch,
        "val_nat_acc": nat_acc,
        "val_query_acc": query_acc
    }, name)
def prune_step(mask: torch.Tensor, prune_num: int ,loader,model):
        feats_list = []
        for data in loader:
            _input, _label = data
            _input, _label = _input.cuda(), _label.cuda()
            _,_feats = model(_input)
            _feats = _feats.abs()
            if _feats.dim() > 2:
                _feats = _feats.flatten(2).mean(2)
            feats_list.append(_feats)
        feats_list = torch.cat(feats_list).mean(dim=0)
        idx_rank = feats_list.argsort()
        counter = 0
        for idx in idx_rank:
            if mask[idx].norm(p=1) > 1e-6:
                mask[idx] = 0.0
                counter += 1
                #print(f'    {output_iter(counter, prune_num)} Prune {idx:4d} / {len(idx_rank):4d}')
                if counter >= min(prune_num, len(idx_rank)):
                    break
def extraction(args, output_dir):
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
        extract_model_archive = mnist.models
        # exit()
    elif args.dataset == 'cifar10':
        query_size = CIFAR_QUERY_SIZE
        model_archive = cifar10.models
        
        if args.dataset_source == 'cifar100':
            train_loader, _, _ = get_cifar100_loaders()
        
        elif args.dataset_source == 'cifar10':
            _,_, train_loader, _, _ = get_cifar10_loaders()
        
        _, _,_,valid_loader, test_loader = get_cifar10_loaders()
        
        extract_model_archive = resnet.models
    
    elif args.dataset == 'cifar100':
        query_size = CIFAR_QUERY_SIZE
        model_archive = cifar10.models_cifar100
        
        if args.dataset_source == 'cifar100':
            
            _,_, train_loader, _, _  = get_cifar100_loaders()
        
        elif args.dataset_source == 'cifar10':
            
            train_loader, _, _ = get_cifar10_loaders()
        
        _, _,_,valid_loader, test_loader = get_cifar100_loaders()

        extract_model_archive = resnet.models
    
    elif args.dataset == 'svhn':
        query_size = CIFAR_QUERY_SIZE
        model_archive = resnet.models
        if args.dataset_source == 'svhn':
            train_loader, _, _ = get_svhn_loaders()
        elif args.dataset_source == 'cifar10':
            train_loader, _, _ = get_cifar10_loaders()
        _, valid_loader, test_loader = get_svhn_loaders()
        extract_model_archive = resnet.models
    elif args.dataset == 'tinyimagenet':
        query_size = TINY_QUERY_SIZE
        model_archive = resnet.models
        train_loader, valid_loader, test_loader = get_tinyimagenet_loaders(batch_size=32)
        extract_model_archive = resnet.models

    resume = os.path.join(output_dir, "../", "checkpoints", "checkpoint_nat_best.pt")
    d = torch.load(resume)
    logger.info(f"logging model checkpoint {d['epoch']}...")
    
    num_classes = 100 if args.dataset == 'cifar100' else 10
    num_classes = 200 if args.dataset == 'tinyimagenet' else num_classes
    if args.model_type=='vgg11':
        model=vgg.VGG('VGG11')
    else:
        model = extract_model_archive[args.model_type](num_classes=num_classes)
    teacher = model_archive[d['model']['type']](num_classes=num_classes)
    teacher.load_state_dict(d['model']['state_dict'])
    model = copy.deepcopy(teacher)
    if args.query_type not in ['adapmixuploc', 'adapmixupquery']:
        query = queries.queries[d['query_model']['type']](query_size=(args.num_query, *query_size),
                                    response_size=(args.num_query,), query_scale=255, response_scale=num_classes)
    else:
        query = queries.queries[d['query_model']['type']](mixup_num = args.num_mixup, query_size=(args.num_query, *query_size),
                                    response_size=(args.num_query,), query_scale=255, response_scale=num_classes)
    
    query.load_state_dict(d['query_model']['state_dict'], strict=False)

    model.to(args.device)
    teacher.to(args.device)
    query.to(args.device)
    teacher.eval()

    if args.dataset == 'mnist':
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        lr_scheduler = None
    elif args.dataset in ['cifar10', 'cifar100']:
        opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
        lr_scheduler = lambda t: np.interp([t],\
            [0, 100, 100, 150, 150, 200],\
            [0.1, 0.1, 0.01, 0.01, 0.001, 0.001])[0]
    elif args.dataset == 'svhn':
        opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
        lr_scheduler = lambda t: np.interp([t],\
            [0, 100, 100, 150, 150, 200],\
            [0.1, 0.1, 0.01, 0.01, 0.001, 0.001])[0]

    best_val_nat_acc = 0
    best_val_nat_query_acc = 0
    best_val_query_acc = 0
    best_val_query_nat_acc = 0
    best_test_nat_acc = 0
    best_test_query_acc = 0

    for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d):
                last_conv: nn.Conv2d = prune.identity(module, 'weight')
                break
    length = last_conv.out_channels

    mask: torch.Tensor = last_conv.weight_mask
    prune_step(mask,max(length*0.95 - 10, 0),valid_loader,model)
    for i in range(min(10, length)):
        print('Iter: ',i )
        prune_step(mask, 1,valid_loader,model)
        val_meters, val_q_meters = extract_loop(model, teacher, query, valid_loader,
            opt, lr_scheduler, i, logger, output_dir,
            max_epoch=args.epoch, mode='val', device=args.device)
        if  val_meters['teach acc']-  val_meters['nat acc'] > 0.2:
            break
    query.eval()
    for epoch in range(args.epoch):
        model.train()
        train_meters, train_q_meters = extract_loop(model, teacher, query, train_loader,
                opt, lr_scheduler, epoch, logger, output_dir,
                max_epoch=args.epoch, mode='train', device=args.device)

        with torch.no_grad():
            model.eval()
            query.eval()
            val_meters, val_q_meters = extract_loop(model, teacher, query, valid_loader,
                opt, lr_scheduler, epoch, logger, output_dir,
                max_epoch=args.epoch, mode='val', device=args.device)
            test_meters, test_q_meters = extract_loop(model, teacher, query, test_loader,
                opt, lr_scheduler, epoch, logger, output_dir,
                max_epoch=args.epoch, mode='test', device=args.device)

            if not os.path.exists(os.path.join(output_dir, "checkpoints")):
                os.makedirs(os.path.join(output_dir, "checkpoints"))
            
            if (epoch+1) % 25 == 0:
                save_extract_ckpt(model, args.model_type, opt, val_meters['nat acc'], val_q_meters['query acc'], epoch,
                                os.path.join(output_dir, "checkpoints", f"checkpoint_{epoch}.pt"))
            
            if best_val_nat_acc <= val_meters['nat acc']:
                save_extract_ckpt(model, args.model_type, opt, val_meters['nat acc'], val_q_meters['query acc'], epoch,
                                os.path.join(output_dir, "checkpoints", f"checkpoint_nat_best.pt"))
                best_val_nat_acc = val_meters['nat acc']
                best_val_nat_query_acc = val_q_meters['query acc']
                best_test_nat_acc = test_meters['nat acc']
                best_test_query_acc = test_q_meters['query acc']
                
            if best_val_query_acc <= val_q_meters['query acc']:
                save_extract_ckpt(model, args.model_type, opt, val_meters['nat acc'], val_q_meters['query acc'], epoch,
                                os.path.join(output_dir, "checkpoints", f"checkpoint_query_best.pt"))
                best_val_query_acc = val_q_meters['query acc']
                best_val_query_nat_acc = val_meters['nat acc']

            save_extract_ckpt(model, args.model_type, opt, val_meters['nat acc'], val_q_meters['query acc'], epoch,
                            os.path.join(output_dir, "checkpoints", f"checkpoint_latest.pt"))
            
    logger.info("="*100)
    logger.info("Best valid query acc : {:.4f}".format(best_val_nat_query_acc))
    logger.info("Best valid nat acc   : {:.4f}".format(best_val_nat_acc))
    logger.info("Best query acc : {:.4f}".format(best_test_query_acc))
    logger.info("Best nat acc   : {:.4f}".format(best_test_nat_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='sanity check for distillation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--exp-name',
        type=str,
        default='experiments')
    parser.add_argument("-dt", "--dataset",
        type=str,
        default='cifar10',
        choices=['mnist', 'cifar10', 'svhn', 'cifar100', 'tinyimagenet'])
    parser.add_argument("-dmt", "--model-type",
        type=str,
        help='distillation model type',
        default='res18',
        choices=['vgg11','big', 'small', 'bigg', 'smallg', 'res18', 'res34', 'res50', 'res101', 'res152'])
    parser.add_argument('-tmt', '--teacher-model-type',
        type=str,
        help='teacher model type',
        default='res18',
        choices=['big', 'small', 'bigg', 'smallg', 'res18', 'res34', 'res50', 'res101', 'res152'])
    parser.add_argument("-tt", "--train-type",
        type=str,
        default='none',
        help='train type, none: no watermark, ood: rand-noise watermark, base: baseline for ind watermark',
        choices=['randomsmooth','none', 'base', 'minmaxpgd', 'minmaxpgd-itervar', 'minmaxpgd-coefvar'])
    parser.add_argument("-dsrc", "--dataset-source",
        type=str,
        default='cifar10',
        choices=['cifar100', 'cifar10', 'svhn'])
    parser.add_argument('-msg', '--message',
        type=str,
        help='additional message for naming the exps.',
        default='')
    parser.add_argument('-admsg', '--additional-message',
        type=str,
        default='')
    parser.add_argument('-qt', '--query-type',
        type=str,
        default='')
    parser.add_argument('-nq', "--num-query",
        type=int,
        help='# of queries',
        default=100)
    parser.add_argument('-nm', "--num-mixup",
        type=int,
        help='# of mixup',
        default=1)
    parser.add_argument('-ep', "--epoch",
        type=int,
        default=100,
        required=False)
    parser.add_argument("--device",
        default='cuda')
    parser.add_argument("--seed",
        type=int,
        default=0)
    parser.add_argument('--labelmode',type=str,default='soft')

    #waitGPU.wait(gpu_ids=[0,1,2,3,4,5,6,7], nproc=0, interval=60)

    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    if args.dataset == 'mnist':
        assert args.model_type in ['big', 'small', 'bigg', 'smallg']
    elif args.dataset == 'cifar10':
        assert args.model_type in ['vgg11','res18', 'res34', 'res50', 'res101', 'res152']
        
    if args.query_type == '':
        exp_name = "_".join([args.dataset, args.teacher_model_type, args.train_type, str(args.num_query), args.message])
    else:
        exp_name = "_".join([args.dataset, args.teacher_model_type, args.query_type, args.train_type, str(args.num_query), str(args.num_mixup), args.message])


    if args.labelmode=='hard':

        output_dir = os.path.join(args.exp_name, exp_name, 'fineprune_hard_label'+args.additional_message)
    
    elif args.labelmode=='soft':

        output_dir = os.path.join(args.exp_name, exp_name, 'fineprune'+args.additional_message)
  

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for s in ['images', 'checkpoints']:
        extra_dir = os.path.join(output_dir, s)
        if not os.path.exists(extra_dir):
            os.makedirs(extra_dir)
    
    extraction(args, output_dir)
