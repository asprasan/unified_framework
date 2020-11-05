'''
-----------------------------------
DEFINE C2B SENSOR FOR BUCKET IMAGES
-----------------------------------
'''
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import math


class Binarize(torch.autograd.Function):

    @staticmethod
    def forward(cxt, input):
        output = input.new_empty(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input



class C2B_trainable(nn.Module):
    
    def __init__(self, block_size, sub_frames, two_bucket=False):
        super(C2B_trainable, self).__init__()

        weight = torch.empty(1, sub_frames, block_size, block_size).cuda()
        weight = weight.bernoulli_(p=0.2).mul_(2).add_(-1).div_(100)
        self.weight = nn.Parameter(weight, requires_grad=True)
        # stdv = math.sqrt(1.5 / (block_size * block_size * sub_frames))
        # self.weight.data.uniform_(-stdv, stdv)
        # self.weight.lr_scale = 1. / stdv
        # nn.init.kaiming_normal_(self.weight)
        self.continuous = self.weight.data.clone()
        
        self.block_size = block_size 
        self.two_bucket = two_bucket
        self.binarize_mask()


    def binarize_mask(self):
        self.binary = Binarize.apply(self.continuous).add_(1).div_(2)
        self.weight.data.copy_(self.binary)


    def forward(self, x):
        _,_,H,W = x.size()
        code_repeat = self.weight.repeat(1, 1, H//self.block_size, W//self.block_size)
        b1 = torch.sum(code_repeat*x, dim=1, keepdim=True) / torch.sum(code_repeat, dim=1, keepdim=True)
        if not self.two_bucket:
            return b1
        code_repeat_comp = 1 - code_repeat
        b0 = torch.sum(code_repeat_comp*x, dim=1, keepdim=True) / torch.sum(code_repeat_comp, dim=1, keepdim=True)
        # b0 = torch.mean(x, dim=1, keepdim=True)
        return b1, b0 # (N,1,H,W)



class C2B(nn.Module):
    
    def __init__(self, block_size, sub_frames, mask='random', two_bucket=False):
        super(C2B, self).__init__()

        if mask == 'impulse':
            assert block_size**2 == sub_frames
            code = torch.eye(block_size**2).cuda()
            code = code.reshape(1, sub_frames, block_size, block_size)
            print('Initialized sensor with impulse code %dx%dx%d'%(sub_frames, block_size, block_size))
        
        elif mask == 'opt':
            ## eccv optimal code 16x8x8
            filename = '/data/prasan/anupama/dataset/eccv18/optimized_SBE_8x8x16'
            code = scipy.io.loadmat(filename)['x']
            code = code.transpose(2,0,1)
            assert code.shape == (sub_frames, block_size, block_size)
            code = torch.cuda.FloatTensor(code).unsqueeze(0)   
            print('Initialized sensor with optimized code from %s'%filename)

        elif mask == 'flutter':
            filename = '/data/prasan/anupama/dataset/flutter_shutter_%dx.npy'%sub_frames
            code = np.load(filename)
            code = torch.cuda.FloatTensor(code).unsqueeze(-1).unsqueeze(-1)
            code = code.repeat(1, block_size, block_size)
            assert code.shape == (sub_frames, block_size, block_size)
            print('Initialized sensor with flutter shutter code from %s'%filename)
        
        else:
            ## random code 16x8x8
            code = torch.empty(1, sub_frames, block_size, block_size).cuda()
            code = code.bernoulli_(p=0.8)
            print('Initialized sensor with random code %dx%dx%d'%(sub_frames, block_size, block_size))
        
        self.block_size = block_size
        self.code = nn.Parameter(code, requires_grad=False)
        self.two_bucket = two_bucket
        # self.code_repeat = code.repeat(1, 1, patch_size//block_size, patch_size//block_size)


    def forward(self, x):
        _,_,H,W = x.size()
        code_repeat = self.code.repeat(1, 1, H//self.block_size, W//self.block_size)
        b1 = torch.sum(code_repeat*x, dim=1, keepdim=True) / torch.sum(code_repeat, dim=1, keepdim=True)
        if not self.two_bucket:
            return b1
        code_repeat_comp = 1 - code_repeat
        # b0 = torch.sum(code_repeat_comp*x, dim=1, keepdim=True) / torch.sum(code_repeat_comp, dim=1, keepdim=True)
        b0 = torch.mean(x, dim=1, keepdim=True)
        return b1, b0 # (N,1,H,W)
