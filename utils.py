'''
-----------------------
DEFINE HELPER FUNCTIONS
-----------------------
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import scipy.misc
from matplotlib import cm
import matplotlib.pyplot as plt
from math import exp
import matplotlib.pyplot as plt

def gradx(tensor):
    return tensor[:,:,:,2:] - tensor[:,:,:,:-2]

def grady(tensor):
    return tensor[:,:,2:,:] - tensor[:,:,:-2,:]


def gradxy(tensor):
    # grad_filter = np.zeros((1,1,3,3))
    filt = np.array([[0.,-1.,0.],[0.,2.,-1.],[0.,0.,0.]])/2.
    # grad_filter[0,0,...] = filt
    grad_filter = torch.cuda.FloatTensor(filt).unsqueeze(0).unsqueeze(0)
    # grad_filter = grad_filter.repeat(tensor.shape[1], 1, 1, 1)
    # grad = F.conv2d(tensor, grad_filter, padding=1, stride=1, groups=tensor.shape[1])
    N,S,H,W = tensor.size()
    tensor = tensor.view(N*S,1,H,W)
    grad = F.conv2d(tensor, grad_filter, padding=1, stride=1)
    grad = grad.view(N,S,H,W)
    return grad


def weighted_L2loss(pred, target):
    grad_weight = gradxy(target).abs() + 1.
    loss = ((pred - target)*grad_weight).pow(2).mean()
    return loss


def weighted_L1loss(pred, target):
    grad_weight = gradxy(target).abs() + 1.
    loss = ((pred-target)*grad_weight).abs().mean()
    return loss



def impulse_inverse(img, block_size):
    # img (N,1,H,W)
    vec_filter = np.eye(block_size**2)
    vec_filter = vec_filter.reshape(block_size**2, block_size, block_size)
    vec_filter = torch.cuda.FloatTensor(vec_filter).unsqueeze(1)
    out = F.conv2d(img, vec_filter, stride=block_size) # (N,d**2,H/d,W/d)
    out = F.upsample(out, scale_factor=block_size, mode='bicubic', align_corners=True)
    return out




def compute_psnr(vid1, vid2):
    # (N,9,H,W)
    assert vid1.shape == vid2.shape
    mse = ((vid1 - vid2)**2).mean(dim=[1,2,3])
    psnr = (20.*torch.log10(1./torch.sqrt(mse))).sum()
    return psnr


def compute_ssim(vid1, vid2):
    # imput video tensors (N,9,H,W)
    # return ssim value sum over batch
    assert vid1.shape == vid2.shape
    sigma = 1.5
    window_size = 11
    channel = vid1.shape[1]

    gauss = torch.cuda.FloatTensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    gauss = gauss/gauss.sum()

    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.cuda()
   
    if torch.max(vid1) > 128:
        max_val = 255
    else:
        max_val = 1

    if torch.min(vid1) < -0.5:
        min_val = -1
    else:
        min_val = 0
    L = max_val - min_val

    padd = 0

    mu1 = F.conv2d(vid1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(vid2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(vid1 * vid1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(vid2 * vid2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(vid1 * vid2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    # cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    # ret = ssim_map.mean(1).mean(1).mean(1)
    ret = ssim_map.mean(dim=[1,2,3]).sum()
    # print(ret.shape)
    return ret



def create_dirs(save_path):
    """create 
       save_path/logs/
       save_path/model/
    """
    if not os.path.exists(os.path.join(save_path)):
        os.mkdir(os.path.join(save_path))
    # if not os.path.exists(os.path.join(save_path,'images')):
    #     os.mkdir(os.path.join(save_path,'images'))
    # if not os.path.exists(os.path.join(save_path,'gifs')):
    #     os.mkdir(os.path.join(save_path,'gifs'))
    if not os.path.exists(os.path.join(save_path,'logs')):
        os.mkdir(os.path.join(save_path,'logs'))
    if not os.path.exists(os.path.join(save_path,'model')):
        os.mkdir(os.path.join(save_path,'model'))
    return


def save_checkpoint(state, save_path, filename):
    torch.save(state, os.path.join(save_path, filename))
    return


def read_image(path, height_img=None, width_img=None):
    # img = scipy.misc.imread(path, mode='L') # grayscale
    # img = scipy.misc.imresize(img, size=(height_img,width_img))
    img = Image.open(path).convert(mode='L')
    if height_img and width_img:
        img = img.resize((width_img,height_img), Image.ANTIALIAS)
    img = np.array(img)
    img = img/255.
    return img


def save_image(img, path, normalize=False):
    # img = scipy.misc.toimage(img, cmin=np.amin(img), cmax=np.amax(img))
    # print(np.amin(img), np.amax(img))
    if normalize:
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    img = img.clip(0,1)
    img = Image.fromarray((img*255.).astype('uint8'))
    img.save(path)
    return


def save_gif(arr, path, normalize=False):
    # (9,H,W)
    frames = []
    for sub_frame in range(arr.shape[0]):
        img = arr[sub_frame,...]
        # print(np.amax(img), np.amin(img))
        if normalize:
            img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
        img = img.clip(0,1)
        img = Image.fromarray((img*255.).astype('uint8'))
        frames.append(img)
    frame1 = frames[0]
    frame1.save(path, save_all=True, append_images=frames[1:], duration=500, loop=0)
    return


def save_map(arr, path, normalize=False):
    # (9,H,W)
    # frames = []
    # cm = plt.get_cmap('jet')
    # for sub_frame in range(arr.shape[0]):
    #     img = arr[sub_frame,...]
    #     # print(np.amax(img), np.amin(img))
    #     if normalize:
    #         img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    #     img = img.clip(0,1)
    #     img = cm(img)
    #     img = Image.fromarray((img[:,:,:3]*255.).astype('uint8'))
    #     frames.append(img)
    # frame1 = frames[0]
    # frame1.save(path, save_all=True, append_images=frames[1:], duration=500, loop=0)
    # return

    img = arr.clip(0,1)
    plt.imsave(path, img, cmap='jet')