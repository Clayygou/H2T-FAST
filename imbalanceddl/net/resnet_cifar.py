'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):
    # if norm_type == 'bn':
    #     norm_dims = [0, 2, 3]
    # elif norm_type == 'in':
    #     norm_dims = [2, 3]
    # elif norm_type == 'ln':
    #     norm_dims = [1, 2, 3]
    # elif norm_type == 'pono':
    #     norm_dims = [1]

    def __init__(self, block, num_blocks, num_classes=10, dim = 1, pono=True, ms=True, re_m_s=True):
        super(ResNet_s, self).__init__()
        self.in_planes = 16


        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # if use_norm:
        #     self.linear = NormedLinear(64, num_classes)
        # else:
        #     self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

        self.pono = PONO(affine=False) if pono else None
        self.ms = MomentShortcut() if ms else None


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    # pono 第一层
    def forward(self, x,**kwargs):
        out_0 = F.relu(self.bn1(self.conv1(x)))

        if 'input2' in kwargs and kwargs['layer'] =='layer0':
            # print(kwargs['input2'])
            out2 = F.relu(self.bn1(self.conv1(kwargs['input2'])))
            out, _, _ = self.pono(out_0,lam_image =kwargs['lam_image'],dim = kwargs['dim'])

            out2, mean, std = self.pono(out2,lam_image = kwargs['lam_image'],dim = kwargs['dim'])
            out_0 = self.ms(out, mean, std)

        out = self.layer1(out_0)
        if 'input2' in kwargs and kwargs['layer'] =='layer1':
            # print(kwargs['input2'])
            out2 = self.layer1(F.relu(self.bn1(self.conv1(kwargs['input2']))))
            out, _, _ = self.pono(out,lam_image =kwargs['lam_image'],dim = kwargs['dim'])

            out2, mean, std = self.pono(out2,lam_image = kwargs['lam_image'],dim = kwargs['dim'])
            out = self.ms(out, mean, std)


        out = self.layer2(out)
        if 'input2' in kwargs and kwargs['layer'] =='layer2':
            # print(kwargs['input2'])
            out2 =self.layer2(self.layer1(F.relu(self.bn1(self.conv1(kwargs['input2'])))))
            out, _, _ = self.pono(out,lam_image =kwargs['lam_image'],dim = kwargs['dim'])

            out2, mean, std = self.pono(out2,lam_image = kwargs['lam_image'],dim = kwargs['dim'])
            out = self.ms(out, mean, std)

        out = self.layer3(out)

        if 'input2' in kwargs and kwargs['layer'] =='layer3':
            out2 =self.layer3(self.layer2(self.layer1(F.relu(self.bn1(self.conv1(kwargs['input2']))))))
            out, _, _ = self.pono(out,lam_image =kwargs['lam_image'],dim = kwargs['dim'])

            out2, mean, std = self.pono(out2,lam_image = kwargs['lam_image'],dim = kwargs['dim'])
            out = self.ms(out, mean, std)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out

class PONO(nn.Module):
    def __init__(self, input_size=None, return_stats=False, affine=True, dim=1, eps=1e-5):
        super(PONO, self).__init__()
        self.return_stats = return_stats
        self.input_size = input_size
        self.eps = eps
        self.affine = affine

        if affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, *input_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, *input_size))
        else:
            self.beta, self.gamma = None, None

    def forward(self, x, **kwargs):
        print('dim ***********',kwargs['dim'])
        mean = x.mean(dim=kwargs['dim'], keepdim=True)
        std = (x.var(dim=kwargs['dim'], keepdim=True) + self.eps).sqrt()
        x = (x - kwargs['lam_image']*mean) / kwargs['lam_image']*std
        if self.affine:
            x = x * self.gamma + self.beta
        return x, kwargs['lam_image']*mean, kwargs['lam_image']*std


class MomentShortcut(nn.Module):
    def __init__(self, beta=None, gamma=None):
        super(MomentShortcut, self).__init__()
        self.gamma, self.beta = gamma, beta

    def forward(self, x, beta=None, gamma=None):
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma
        if gamma is not None:
            x.mul_(gamma)
        if beta is not None:
            x.add_(beta)
        return x

def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet_s(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()