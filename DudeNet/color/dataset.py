import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
import torch
import torch.nn as nn
from utils import data_augmentation

import pandas as pd #23

def normalize(data):
    return data/255.

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    #print endc
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def prepare_data(data_path, patch_size, stride, aug_times=1):
    # train
    print('process training data')
    scales = [1, 0.9, 0.8, 0.7]
    #files = glob.glob(os.path.join(data_path, 'train', '*'))
    files = glob.glob(os.path.join(data_path, 'CTrain400', '*'))
    files.sort()
    h5f = h5py.File('train.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            #print Img.shape
            Img = torch.tensor(Img)
            Img = Img.permute(2,0,1)
            Img = Img.numpy()
            #Img = np.expand_dims(Img[:,:,0].copy(), 0)
            #Img= np.transpose(image, (2,1,0))
            #print  Img.shape
            Img = np.float32(normalize(Img))
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    #files.clear()
    files = glob.glob(os.path.join(data_path, 'CBSD68', '*.bmp'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        #img = np.expand_dims(img[:,:,0], 0) #201810271714tcw
        img = torch.tensor(img)
        img = img.permute(2,0,1)
        img = img.numpy()
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)

class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        #breakpoint()
        if self.train:
            #h5f = pd.HDFStore('/home/bbb/23wmcgcnn/DudeNet/color/train.h5', "r")
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(h5f.keys())
        #self.keys = list(np.arange(200))
        random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        #data = np.array(np.ones((3,32,32)))
        h5f.close()
        return torch.Tensor(data)