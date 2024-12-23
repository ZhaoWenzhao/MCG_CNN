import numpy as np 
from scipy import special
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Change based on environment
path_to_bessel = "/home/bbb/23wmcgcnn/ConvNeXt/models/bessel.npy"#"./bessel.npy"

def cartesian_to_polar_coordinates(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (phi, rho)


def calculate_FB_bases_shear(L1, alpha, rot_theta, shear, maxK):
    '''
    s = 2^alpha is the scale
    alpha <= 0
    maxK is the maximum num of bases you need
    '''
    maxK = np.min([(2 * L1 + 1)**2-1, maxK])

    L = L1 + 1
    R = L1 + 0.5

    truncate_freq_factor = 2.5

    if L1 < 2:
        truncate_freq_factor = 2.5

    xx, yy = np.meshgrid(range(-L, L+1), range(-L, L+1))
    xx = xx+shear*yy
    
    xx = alpha*xx/(R)
    yy = alpha*yy/(R)

    ugrid = np.concatenate([yy.reshape(-1,1), xx.reshape(-1,1)], 1)
    # angleGrid, lengthGrid
    tgrid, rgrid = cartesian_to_polar_coordinates(ugrid[:,0], ugrid[:,1])

    tgrid = tgrid+rot_theta 

    num_grid_points = ugrid.shape[0]

    maxAngFreq = 15
    
    # change path based on environment
    bessel = np.load(path_to_bessel)

    B = bessel[(bessel[:,0] <= maxAngFreq) & (bessel[:,3]<= np.pi*R*truncate_freq_factor)]

    idxB = np.argsort(B[:,2])

    mu_ns = B[idxB, 2]**2

    ang_freqs = B[idxB, 0]
    rad_freqs = B[idxB, 1]
    R_ns = B[idxB, 2]

    num_kq_all = len(ang_freqs)
    max_ang_freqs = max(ang_freqs)

    Phi_ns=np.zeros((num_grid_points, num_kq_all), np.float32)

    Psi = []
    kq_Psi = []
    num_bases=0

    for i in range(B.shape[0]):
        ki = ang_freqs[i]
        qi = rad_freqs[i]
        rkqi = R_ns[i]

        r0grid=rgrid*R_ns[i]

        F = special.jv(ki, r0grid)

        Phi = 1./np.abs(special.jv(ki+1, R_ns[i]))*F

        Phi[rgrid >=1]=0

        Phi_ns[:, i] = Phi

        if ki == 0:
            Psi.append(Phi)
            kq_Psi.append([ki,qi,rkqi])
            num_bases = num_bases+1

        else:
            Psi.append(Phi*np.cos(ki*tgrid)*np.sqrt(2))
            Psi.append(Phi*np.sin(ki*tgrid)*np.sqrt(2))
            kq_Psi.append([ki,qi,rkqi])
            kq_Psi.append([ki,qi,rkqi])
            num_bases = num_bases+2
                        
    Psi = np.array(Psi)
    kq_Psi = np.array(kq_Psi)

    num_bases = Psi.shape[1]

    if num_bases > maxK:
        Psi = Psi[:maxK]
        kq_Psi = kq_Psi[:maxK]
    num_bases = Psi.shape[0]
    p = Psi.reshape(num_bases, 2*L+1, 2*L+1).transpose(1,2,0)
    psi = p[1:-1, 1:-1, :]

    psi = psi.reshape((2*L1+1)**2, num_bases)
        
    # normalize
    # using the sum of psi_0 to normalize.
    c = np.sum(psi[:,0])
    
    # psi.shape example: (9, 6), (25, 6)
    # psi.shape: (filter_map^2, num_basis)
    psi = psi/c

    return psi, c, kq_Psi

def tensor_fourier_bessel_affine(c_o,c_in,size, rotation_up, rotation_low, scale_up, scale_low, shear_limit, num_funcs=None):
    '''
    Basis of fourier bessel.
    (rotation_low,rotation_up)->(-1*np.pi,1*np.pi)
    scale -> (1.0,3.5)
    shear_limit -> (-0.5,0.5)    
    '''
    base_rotation = np.random.rand(c_o,c_in)*(rotation_up-rotation_low)+rotation_low
    base_scale = np.random.rand(c_o,c_in)*(scale_up-scale_low)+scale_low
    base_shear = (np.random.rand(c_o,c_in)*2-1)*shear_limit
    base_shear = np.tan(np.pi*base_shear)
    
    base_rotation = np.tile(base_rotation[:,:,None,None],(1,1,size,size))
    base_scale = np.tile(base_scale[:,:,None,None],(1,1,size,size))
    base_shear = np.tile(base_shear[:,:,None,None],(1,1,size,size))
    
    max_order = size-1
    
    num_funcs = num_funcs or size ** 2
    num_funcs_per_scale = ((max_order + 1) * (max_order + 2)) // 2

    basis_xy = []

    bxy = []
    
    for i in range(c_o):
        for j in range(c_in):

            psi, c, kq_Psi = calculate_FB_bases_shear(size//2, base_scale[i,j,0,0], base_rotation[i,j,0,0], base_shear[i,j,0,0], num_funcs)
            psi=psi.transpose((1,0))
            # Add zero frequency basis ################
            base_n_m=np.concatenate((psi,1.0/size/size*np.ones((1,size*size))),axis=0).reshape((-1,size,size))
             
            bxy.append(base_n_m)

    basis_xy.extend(bxy)

    basis = torch.Tensor(np.stack(basis_xy))#[:num_funcs]

    basis = basis.reshape(c_o,c_in,num_funcs,size, size).permute((2,0,1,3,4)).contiguous()

    return basis


'''
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
'''

class Conv2dstr_fb(nn.Module):
    """
    Convolution with augmented Fourier Bessel filters
    """
    '''
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    '''
    def __init__(self, 
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = 'zeros',
        numbasis: int = -1,
        mode: str = 'hiddenlayer'):
        super(Conv2dstr_fb, self).__init__()
        self.kernel_size=kernel_size
        self.patch_x=self.kernel_size
        self.patch_y=self.kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.dilation = dilation
        self.mode = mode
        
        if numbasis == -1:
            self.numbasis = kernel_size*kernel_size
        else:
            self.numbasis = numbasis
        self.groups = groups

        if bias is False:
            self.bias = None 
        else:
            self.bias = bias
        self.padding_mode = padding_mode            

        if self.mode == 'hiddenlayer':

            self.register_buffer('wt_filter', self.get_str_fb_filter_tensor(self.numbasis,
                self.in_channels//self.groups,self.out_channels,self.patch_x,self.patch_y))  
            self.weight = torch.nn.Parameter(torch.Tensor(self.numbasis,self.out_channels,self.in_channels//self.groups))
            torch.nn.init.xavier_uniform_(
            self.weight,
            gain=torch.nn.init.calculate_gain("linear"))
        else:

            self.register_buffer('wt_filter', self.get_str_fb_filter_tensor(self.numbasis,
            kernel_size*kernel_size*self.in_channels//self.groups,self.out_channels,self.patch_x,self.patch_y))
            self.weight = torch.nn.Parameter(torch.Tensor(self.numbasis,self.out_channels,
            kernel_size*kernel_size*self.in_channels//self.groups))
            torch.nn.init.xavier_uniform_(
            self.weight,
            gain=torch.nn.init.calculate_gain("linear"))                        
        
        
    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        b,c,h,w = x.shape

        ##b, t, h, w = x.shape
        filter = torch.einsum('nct,nctij->nctij', self.weight, self.wt_filter).contiguous().mean(dim=0).contiguous()
        if self.mode == 'hiddenlayer':
            result = torch.nn.functional.conv2d(x,filter, stride=self.stride, padding=(self.padding,self.padding), dilation=self.dilation, groups=self.groups, bias=self.bias)
        else:
            x = x.repeat(1,self.kernel_size*self.kernel_size,1,1)    
            result = torch.nn.functional.conv2d(x,filter, stride=self.stride, padding=(self.padding,self.padding), dilation=self.dilation, groups=self.groups, bias=self.bias)
        
        return result 

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if self.numbasis != -1:
            s += ', numbasis={numbasis}'
        return s.format(**self.__dict__)
    
    def get_str_fb_filter_tensor(self, numbasis, c_in, channel, patch_x, patch_y,seeds=20):
        c_o=channel
        c_in=c_in
        size=patch_x
        rotation_up=np.pi
        rotation_low=-1*np.pi
        scale_up=2.0   
        scale_low=1
        shear_limit=0.25 
        np.random.seed(seeds)
        wt_filter=tensor_fourier_bessel_affine(c_o,c_in,size, rotation_up, rotation_low, scale_up, scale_low, shear_limit)
        wt_filter = wt_filter[0:numbasis,:,:,:,:]
        return torch.tensor(wt_filter,dtype=torch.float)
    
