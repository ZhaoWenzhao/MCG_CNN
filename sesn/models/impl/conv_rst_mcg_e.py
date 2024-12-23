import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from .ses_basis_rst import steerable_A, steerable_A1, steerable_B, steerable_C, steerable_D, steerable_D1, steerable_E, steerable_F, steerable_G, steerable_G1, steerable_H, steerable_I1
from .ses_basis_rst import normalize_basis_by_min_scale, normalize_basis_by_mean_scale, normalize_basis_dot

import numpy as np

class Conv_MCG_E(nn.Module):
    '''
        Convolution with weighted augmented filters
    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        effective_size: The effective size of the kernel with the same # of params
        scales: List of scales of basis
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        bias: If ``True``, adds a learnable bias to the output
    '''

    def __init__(self, in_channels, out_channels, kernel_size, effective_size,
                 scales=np.array([1.0]), rotations = np.array([0.0]), stride=1, padding=0, groups=1, bias=False, basis_type='E', **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.effective_size = effective_size
        self.scales = scales
            
        n_scales = len(scales)
        n_rot = len(rotations)
        
        self.num_scales = n_scales 
        self.rotations = rotations
        self.num_rotations = n_rot 
        self.stride = stride
        self.padding = padding
        self.groups = groups

        print('E_mcg_rst')
        basis = steerable_E(kernel_size, self.rotations, self.scales, effective_size, **kwargs)

        print(basis.shape)
        basis = normalize_basis_by_min_scale(basis)
        print("Normalized!!")

        seeds = 20

        c_o=self.out_channels
        c_in=self.in_channels//self.groups
        size=kernel_size
        print(seeds)
        np.random.seed(seeds)

        t_ct = np.random.uniform(low=0, high=self.num_rotations*self.num_scales, size=(c_o*c_in)).astype(int)
        basis=basis[:,t_ct,:,:]
        print(basis.shape)
       
        basis = basis.view(-1,self.out_channels, self.in_channels//self.groups, 
                         self.kernel_size, self.kernel_size)
        
        self.register_buffer('basis', basis)

        self.num_funcs = self.basis.size(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels//self.groups, self.num_funcs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):

        kernel = torch.einsum('ctn,nctij->ctij', self.weight, self.basis)

        kernel = kernel.view(self.out_channels, self.in_channels//self.groups, 
                             self.kernel_size, self.kernel_size)
        
        # convolution    
        y = F.conv2d(x, kernel, bias=None, stride=self.stride, padding=self.padding,groups=self.groups)
        # y is in shape [batch, channel x num_rotations x num_scales, height, width], indexed by [batch, lambda theta alpha, u1, u2 ]
        B, C, H, W = y.shape
        y = y.view(B, self.out_channels, #self.num_rotations * self.num_scales, 
                   H, W)
        # y is in shape [batch, channel, num_rotations x num_scales, height, width], indexed by [batch, lambda, theta alpha, u1, u2 ]

        if self.bias is not None:
            y = y + self.bias.view(1, -1, #1,
                                   1, 1)

        return y

    def extra_repr(self):
        s = '{in_channels}->{out_channels} | scales={scales} | size={kernel_size}'
        return s.format(**self.__dict__)

