# Adaptive aggregation of Monte Carlo augmented decomposed filters for efficient group-equivariant convolutional neural network
PyTorch implementation of the paper "[Adaptive aggregation of Monte Carlo augmented decomposed filters for efficient group-equivariant convolutional neural network](https://arxiv.org/abs/2305.10110)".

## Installation

The codes for the training and test neural networks on different datasets for different tasks are borrowed from codes from other papers as listed in the Acknowledgement section.
Therefore, the installation methods are mostly the same. The user should check out the corrsponding GitHub repositories. 
It is recommended to use [PyTorch docker images](https://hub.docker.com/r/pytorch/pytorch/tags) of PyTorch 1.8.1 for the installation.


## Usage

### Equivariant experiments on the RSS-MNIST dataset

#### 1. Data generation
```
cd sesn
sh prepare_mnist_scale_rss.sh
```

#### 2. Training and test ResNeXt50-WMCG-k15
```
sh experiments_mnist_rot_scale_shear.sh
```

### Image classification on ImageNet 
Under the folder ConvNeXt is the codes for classification on ImageNet using ConvNeXt as the base model. To use our convolutional layers, please adjust the path to bessel.npy in the conv2dstr_fb.py file appropriately for your environment.

#### Experiments with ConvNeXt-S-k7-WMCG-nb49 on ImageNet40 
```
cd ConvNeXt
sh launch_training_cls_40.sh
```

#### Experiments with ConvNeXt-S-k7-WMCG-nb49 on ImageNet1k 
```
sh launch_training_cls_1k.sh
```

### Image denoising experiments
To use our convolutional layers, please adjust the path to bessel.npy in the conv2dstr_fb_p1.py file appropriately for your environment.
An example command for training DudeNeXt-k5-WMCG-nb9 for real-noisy image denoising
```
cd DudeNet/real_noisy
python train_r.py --preprocess True --num_of_layers 17 --mode S --noiseL 25 --val_noiseL 25
```

## Acknowledgement

[https://github.com/facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)

[https://github.com/gaoliyao/Roto-scale-translation-Equivariant-CNN](https://github.com/gaoliyao/Roto-scale-translation-Equivariant-CNN)

[https://github.com/hellloxiaotian/DudeNet](https://github.com/hellloxiaotian/DudeNet)