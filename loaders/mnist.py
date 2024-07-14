import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.nn.functional as F



def get_mnist_loaders(root='../data'):
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor()
    ])
    transform_test = transforms.ToTensor()
    dev_set = datasets.MNIST(root=root, train=True, download=True, transform=transform_train)
    test_set = datasets.MNIST(root=root, train=False, download=True, transform=transform_test)
    seed = torch.get_rng_state()
    torch.manual_seed(0)
    train_set, val_set = torch.utils.data.random_split(dev_set, [59000, 1000])
    torch.set_rng_state(seed)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=2, drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        val_set, batch_size=128, shuffle=False, num_workers=2, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=False, num_workers=2, drop_last=False
    )

    return train_loader, valid_loader, test_loader