'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import os
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import json

from .cutout import Cutout


mean = {
    'stl10': (0.4467, 0.4398, 0.4066),
    'scale_mnist': (0.0607,),
    'rot_scale_mnist': (0.0607,),
    'rot_scale_shear_mnist': (0.0607,),
}

std = {
    'stl10': (0.2603, 0.2566, 0.2713),
    'scale_mnist': (0.2161,),
    'rot_scale_mnist': (0.2161,),
    'rot_scale_shear_mnist': (0.2161,),
}


def loader_repr(loader):
    # fix some problems with torchvision 0.3.0 with VisionDataset
    if not isinstance(loader.dataset, ConcatDataset):
        s = ('{dataset.__class__.__name__} Loader: '
             'num_workers={num_workers}, '
             'pin_memory={pin_memory}, '
             'sampler={sampler.__class__.__name__}\n'
             'Root: {dataset.root}\n'
             )
        s = s.format(**loader.__dict__)
        s += 'Data Points: {}\n{}\nTransforms:\n{}'
        s = s.format(len(loader.dataset), loader.dataset.extra_repr(), loader.dataset.transform)
        return s
    else:
        s = ('{dataset.__class__.__name__} Loader: '
             'num_workers={num_workers}, '
             'pin_memory={pin_memory}, '
             'sampler={sampler.__class__.__name__}\n'
             )
        s = s.format(**loader.__dict__)
        for d in loader.dataset.datasets:
            s += '| Dataset {}, Root: {}\n'.format(d.__class__.__name__, d.root)
        s += 'Data Points: {}\n \nTransforms:' + '\n{}' * len(loader.dataset.datasets)
        s = s.format(len(loader.dataset), *[d.transform for d in loader.dataset.datasets])
        return s


#################################################
##################### STL-10 ####################
#################################################
def stl10_plus_train_loader(batch_size, root, download=True):
    transform = transforms.Compose([
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['stl10'], std['stl10']),
        Cutout(1, 32),
    ])
    dataset = datasets.STL10(root=root, transform=transform, download=download)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader


def stl10_test_loader(batch_size, root, download=True):
    transform = transforms.Compose([
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean['stl10'], std['stl10'])
    ])
    dataset = datasets.STL10(root=root, split='test', transform=transform, download=download)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=2)
    return loader


#################################################
##################### SCALE #####################
#################################################
def scale_mnist_train_loader(batch_size, root, extra_scaling=1):
    transform_modules = []
    if not extra_scaling == 1:
        if extra_scaling > 1:
            extra_scaling = 1 / extra_scaling
        scale = (extra_scaling, 1 / extra_scaling)
        print('-- extra scaling ({:.3f} - {:.3f}) is used'.format(*scale))
        scaling = transforms.RandomAffine(0, scale=scale, resample=3)
        transform_modules.append(scaling)

    transform_modules = transform_modules + [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['scale_mnist'], std['scale_mnist'])
    ]

    transform = transforms.Compose(transform_modules)
    dataset = datasets.ImageFolder(os.path.join(root, 'train'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader


def scale_mnist_val_loader(batch_size, root):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['scale_mnist'], std['scale_mnist'])
    ])
    dataset = datasets.ImageFolder(os.path.join(root, 'val'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader


def scale_mnist_test_loader(batch_size, root):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['scale_mnist'], std['scale_mnist'])
    ])
    dataset = datasets.ImageFolder(os.path.join(root, 'test'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=2)
    return loader

#################################################
##################### ROTATION-SCALE #####################
#################################################
def rot_scale_mnist_train_loader(batch_size, root, extra_scaling=1):
    transform_modules = []
    if not extra_scaling == 1:
        if extra_scaling > 1:
            extra_scaling = 1 / extra_scaling
        scale = (extra_scaling, 1 / extra_scaling)
        print('-- extra scaling ({:.3f} - {:.3f}) is used'.format(*scale))
        scaling = transforms.RandomAffine(0, scale=scale, resample=3)
        transform_modules.append(scaling)

    transform_modules = transform_modules + [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['rot_scale_mnist'], std['rot_scale_mnist'])
    ]

    transform = transforms.Compose(transform_modules)
    dataset = datasets.ImageFolder(os.path.join(root, 'train'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader

def rot_scale_mnist_train_loader23(batch_size, root, extra_aug=1):
    transform_modules = []
    if extra_aug == 1:
        #if extra_scaling > 1:
        #    extra_scaling = 1 / extra_scaling
        #scale = (extra_scaling, 1 / extra_scaling)
        #print('-- extra scaling ({:.3f} - {:.3f}) is used'.format(*scale))
        min_scale=0.5
        max_scale=2.0#0.5#1.0
        min_rot =-180 
        max_rot =180
        #scaling = transforms.RandomAffine(0, scale=scale, resample=3)
        transform00 = transforms.RandomAffine([min_rot, max_rot], scale=(min_scale, max_scale), resample=3)
        transform_modules.append(transform00)

    transform_modules = transform_modules + [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['rot_scale_mnist'], std['rot_scale_mnist'])
    ]

    transform = transforms.Compose(transform_modules)
    dataset = datasets.ImageFolder(os.path.join(root, 'train'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader

def rot_scale_mnist_val_loader(batch_size, root):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['rot_scale_mnist'], std['rot_scale_mnist'])
    ])
    dataset = datasets.ImageFolder(os.path.join(root, 'val'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader


def rot_scale_mnist_test_loader(batch_size, root):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['rot_scale_mnist'], std['rot_scale_mnist'])
    ])
    dataset = datasets.ImageFolder(os.path.join(root, 'test'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=2)
    return loader

#################################################
##################### ROTATION-SCALE-SHEAR #####################
#################################################
def rot_scale_shear_mnist_train_loader(batch_size, root, extra_scaling_rot_shear=1):
    #breakpoint()
    #print(extra_scaling_rot_shear)
    #breakpoint()
    transform_modules = []
    if not extra_scaling_rot_shear == 1:
        if extra_scaling_rot_shear > 1:
            extra_scaling_rot_shear = 1 / extra_scaling_rot_shear
        scale = (extra_scaling_rot_shear, 1 / extra_scaling_rot_shear)
        print('-- extra scaling ({:.3f} - {:.3f}) is used'.format(*scale))
        #scaling = transforms.RandomAffine(0, scale=scale, resample=3)
        scaling = transforms.RandomAffine(45, scale=scale, shear=30, resample=3)
        transform_modules.append(scaling)

    transform_modules = transform_modules + [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['rot_scale_shear_mnist'], std['rot_scale_shear_mnist'])
    ]

    transform = transforms.Compose(transform_modules)
    dataset = datasets.ImageFolder(os.path.join(root, 'train'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader

def rot_scale_shear_mnist_train_loader_aug(batch_size, root, extra_aug = 1):#extra_scaling_rot_shear=1):
    
    transform_modules = []
    '''
    if not extra_scaling_rot_shear == 1:
        if extra_scaling_rot_shear > 1:
            extra_scaling_rot_shear = 1 / extra_scaling_rot_shear
        scale = (extra_scaling_rot_shear, 1 / extra_scaling_rot_shear)
        print('-- extra scaling ({:.3f} - {:.3f}) is used'.format(*scale))
        #scaling = transforms.RandomAffine(0, scale=scale, resample=3)
        scaling = transforms.RandomAffine(45, scale=scale, shear=30, resample=3)
        transform_modules.append(scaling)
    '''    
    if extra_aug == 1:
        #if extra_scaling > 1:
        #    extra_scaling = 1 / extra_scaling
        #scale = (extra_scaling, 1 / extra_scaling)
        #print('-- extra scaling ({:.3f} - {:.3f}) is used'.format(*scale))
        min_scale=0.5
        max_scale=2.0#0.5#1.0
        min_rot =-180 #0#-45#
        max_rot =180 #0#45#
        min_shear = -45#0#
        max_shear = 45#0#
        #scaling = transforms.RandomAffine(0, scale=scale, resample=3)
        #transform00 = transforms.RandomAffine([min_rot, max_rot], scale=(min_scale, max_scale), resample=3)
        print("extra_aug !!!!!!")
        transform00 = transforms.RandomAffine([min_rot, max_rot], scale=(min_scale, max_scale), shear=(min_shear,max_shear), resample=3)
        transform_modules.append(transform00)

    transform_modules = transform_modules + [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['rot_scale_shear_mnist'], std['rot_scale_shear_mnist'])
    ]

    transform = transforms.Compose(transform_modules)
    dataset = datasets.ImageFolder(os.path.join(root, 'train'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader

def rot_scale_shear_mnist_train_loader_wrong(batch_size, root, extra_scaling_rot_shear=1):
    
    transform_modules = []
    if not extra_scaling_rot_shear == 1:
        if extra_scaling_rot_shear > 1:
            extra_scaling_rot_shear = 1 / extra_scaling_rot_shear
        scale = (extra_scaling_rot_shear, 1 / extra_scaling_rot_shear)
        print('-- extra scaling ({:.3f} - {:.3f}) is used'.format(*scale))
        #scaling = transforms.RandomAffine(0, scale=scale, resample=3)
        scaling = transforms.RandomAffine(45, scale=scale, shear=0.1, resample=3)
        transform_modules.append(scaling)

    transform_modules = transform_modules + [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['rot_scale_shear_mnist'], std['rot_scale_shear_mnist'])
    ]

    transform = transforms.Compose(transform_modules)
    dataset = datasets.ImageFolder(os.path.join(root, 'train'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader

def rot_scale_shear_mnist_val_loader(batch_size, root):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['rot_scale_shear_mnist'], std['rot_scale_shear_mnist'])
    ])
    dataset = datasets.ImageFolder(os.path.join(root, 'val'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader


def rot_scale_shear_mnist_test_loader(batch_size, root):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['rot_scale_shear_mnist'], std['rot_scale_shear_mnist'])
    ])
    dataset = datasets.ImageFolder(os.path.join(root, 'test'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=2)
    return loader


############################################################################


#################################################
def rot_mnist_train_loader(batch_size, root, extra_rot=1):
    
    transform_modules = []
    if not extra_rot == 1:
        
        print('-- extra rotation ({:.3f} - {:.3f}) is used')
        #scaling = transforms.RandomAffine(0, scale=scale, resample=3)
        rot = transforms.RandomAffine(45, scale=(1.0,1.0), shear=0, resample=3)
        transform_modules.append(rot)

    transform_modules = transform_modules + [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['rot_scale_shear_mnist'], std['rot_scale_shear_mnist'])
    ]

    transform = transforms.Compose(transform_modules)
    dataset = datasets.ImageFolder(os.path.join(root, 'train'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader

def rot_mnist_val_loader(batch_size, root):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['rot_scale_shear_mnist'], std['rot_scale_shear_mnist'])
    ])
    dataset = datasets.ImageFolder(os.path.join(root, 'val'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader


def rot_mnist_test_loader(batch_size, root):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['rot_scale_shear_mnist'], std['rot_scale_shear_mnist'])
    ])
    dataset = datasets.ImageFolder(os.path.join(root, 'test'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=2)
    return loader