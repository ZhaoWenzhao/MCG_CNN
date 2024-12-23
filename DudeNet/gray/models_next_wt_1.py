import torch
import torch.nn as nn
#import pytorch_fft.fft as fft

from conv2dstr_fb_p1 import Conv2dstr_fb_p1 as Conv2dwt

def conv3x3wt(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return Conv2dwt(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BottleneckWT(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=8,
                 base_width=16, dilation=1, norm_layer=None):
        super(BottleneckWT, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3wt(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

###############################################################################################

class Conv_BN_Relu_first(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,groups,bias):
        super(Conv_BN_Relu_first,self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups =1 
        self.conv = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))

class Conv_BN_Relu_other(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,groups,bias):
        super(Conv_BN_Relu_other,self).__init__()
        kernel_size = 3
        padding = 1
        features = out_channels
        groups =1 
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))


class Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,groups,bais):
        super(Conv,self).__init__()
        kernel_size = 3
        padding = 1
        features = 1
        groups =1 
        self.conv = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups, bias=False)
    def forward(self,x):
        return self.conv(x)

class Self_Attn(nn.Module):
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1)
        self.gamma=nn.Parameter(torch.zeros(1))
        self.softmax=nn.Softmax(dim=-1)
    def forward(self,x):
        m_batchsize, C, width,height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize,-1,width*height)
        print(proj_query.size())
        print(proj_key.size())
        print('5')
        energy = torch.bmm(proj_query,proj_key)
        print('6')
        #print energy.size()
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) 
        print('1')
        out = torch.bmm(proj_value,attention.permute(0,2,1))
        print('2')
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + x
        return out, attention

class DudeNeXtWT_1(nn.Module):
    def __init__(self, channels=1, num_of_layers=15):
        super(DudeNeXtWT_1, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups =1 
        layers = []
        kernel_size1 = 1
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_2 = BottleneckWT(features,features,dilation=2) 
        self.conv1_3 = BottleneckWT(features,features) 
        self.conv1_4 = BottleneckWT(features,features) 
        self.conv1_5 = BottleneckWT(features,features,dilation=2) 
        self.conv1_6 = BottleneckWT(features,features) 
        self.conv1_7 = BottleneckWT(features,features) 
        self.conv1_8 = BottleneckWT(features,features) 
        self.conv1_9 = BottleneckWT(features,features,dilation=2) 
        self.conv1_10 = BottleneckWT(features,features) 
        self.conv1_11 = BottleneckWT(features,features) 
        self.conv1_12 = BottleneckWT(features,features,dilation=2) 
        self.conv1_13 = BottleneckWT(features,features) 
        self.conv1_14 = BottleneckWT(features,features) 
        self.conv1_15 = BottleneckWT(features,features) 
        self.conv1_16 = BottleneckWT(features,features) 
        self.conv3 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=1,stride=1,padding=0,groups=1,bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(2*features)
        self.Tanh= nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.conv2_1 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.ReLU(inplace=True))
        self.conv2_2 = BottleneckWT(features,features) 
        self.conv2_3 = BottleneckWT(features,features) 
        self.conv2_4 = BottleneckWT(features,features) 
        self.conv2_5 = BottleneckWT(features,features) 
        self.conv2_6 = BottleneckWT(features,features) 
        self.conv2_7 = BottleneckWT(features,features) 
        self.conv2_8 = BottleneckWT(features,features) 
        self.conv2_9 = BottleneckWT(features,features) 
        self.conv2_10 = BottleneckWT(features,features) 
        self.conv2_11 = BottleneckWT(features,features) 
        self.conv2_12 = BottleneckWT(features,features) 
        self.conv2_13 = BottleneckWT(features,features) 
        self.conv2_14 = BottleneckWT(features,features) 
        self.conv2_15 = BottleneckWT(features,features) 
        self.conv2_16 = nn.Conv2d(in_channels=features,out_channels=features,kernel_size=1,padding=0,groups=groups,bias=False)
        self.conv3_1 = nn.Conv2d(in_channels=2*features,out_channels=1,kernel_size=1,padding=0,groups=groups,bias=False)
        self.conv3_2 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=1,padding=0,groups=groups,bias=False)
      
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)
    def _make_layers(self, block,features, kernel_size, num_of_layers, padding=1, groups=1, bias=False):
        layers = []
        for _ in range(num_of_layers):
            layers.append(block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias))
        return nn.Sequential(*layers)
    def forward(self, x):
        input = x 
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)   
        x1t = self.conv1_8(x1)
        x1 = self.conv1_9(x1t)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        #print x1.size()
        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)
        x2 = self.conv2_3(x2)
        x2 = self.conv2_4(x2)
        x2 = self.conv2_5(x2)
        x2 = self.conv2_6(x2)
        x2 = self.conv2_7(x2)   
        x2 = self.conv2_8(x2)
        x2 = self.conv2_9(x2)
        x2 = self.conv2_10(x2)
        x2 = self.conv2_11(x2)
        x2 = self.conv2_12(x2)
        x2 = self.conv2_13(x2)
        x2 = self.conv2_14(x2)
        x2 = self.conv2_15(x2)
        x2 = self.conv2_16(x2)
        #print x2.size()
        x3 = torch.cat([x1,x2],1)
        x3 = self.BN(x3)
        x3 = self.ReLU(x3)
        x3 = self.conv3_1(x3)
        x4 = torch.cat([x,x3],1)
        x4 = self.conv3_2(x4)
        out = x - x4
        return out
