'''
-----------------------------------
TRAINING CODE - SHIFTVARCONV + UNET
-----------------------------------
'''
import os 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import glob
import argparse
import time
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

## set random seed
torch.manual_seed(12)
np.random.seed(12)


from dataloader import Dataset_load
from sensor import C2B
from unet import UNet
from inverse import ShiftVarConv2D, StandardConv2D
import utils


## parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--expt', type=str, required=True, help='expt name')
parser.add_argument('--save_root',type=str,required=False,default='models', help='root path to save trained models')
parser.add_argument('--epochs', type=int, default=500, help='num epochs to train')
parser.add_argument('--batch', type=int, required=True, help='batch size for training and validation')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--blocksize', type=int, default=8, help='tile size for code default 3x3')
parser.add_argument('--subframes', type=int, default=16, help='num sub frames')
parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to load')
parser.add_argument('--gpu', type=str, required=True, help='GPU ID')
parser.add_argument('--mask', type=str, default='random', help='"impulse" or "random" or "opt" or "flutter"')
parser.add_argument('--two_bucket', action='store_true', help='1 bucket or 2 buckets')
parser.add_argument('--intermediate', action='store_true', help="intermediate reconstruction")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

## params for DataLoader
train_params = {'batch_size': args.batch,
                'shuffle': True,
                'num_workers': 20,
                'pin_memory': True}
val_params = {'batch_size': args.batch,
              'shuffle': False,
              'num_workers': 20,
              'pin_memory': True}


num_epochs = args.epochs

save_path = os.path.join(args.save_root, args.expt)
utils.create_dirs(save_path)


## tensorboard summary logger
writer = SummaryWriter(
        log_dir=os.path.join(save_path, 'logs'))

## configure runtime logging
logging.basicConfig(level=logging.INFO,
                    filename=os.path.join(save_path, 'logs', 'logfile.log'), 
                    format='%(asctime)s - %(message)s', 
                    filemode='w')
# logger=logging.getLogger()#.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logging.getLogger('').addHandler(console)
logging.info(args)



## dataloaders using hdf5 file
data_path = None


try:
    assert data_path is not None
except AssertionError:
    print('path to hdf5 data not specified')
    print('hdf5 data should contain two datasets train and test')
    print('++++exiting++++')
    exit(0)

## initializing training and validation data generators
training_set = Dataset_load(data_path, dataset='train', num_samples='all')
training_generator = data.DataLoader(training_set, **train_params)
logging.info('Loaded training set: %d videos'%(len(training_set)))

validation_set = Dataset_load(data_path, dataset='test', num_samples=60000)
validation_generator = data.DataLoader(validation_set, **val_params)
logging.info('Loaded validation set: %d videos'%(len(validation_set)))



## initialize nets
if args.intermediate:
    num_features = args.subframes
else:
    num_features = 64
c2b = C2B(block_size=args.blocksize, sub_frames=args.subframes, mask=args.mask, two_bucket=args.two_bucket).cuda()
if args.mask == 'flutter':
    assert not args.two_bucket
    invNet = StandardConv2D(out_channels=num_features, window=7).cuda()
else: 
    invNet = ShiftVarConv2D(out_channels=num_features, block_size=args.blocksize, two_bucket=args.two_bucket).cuda()
uNet = UNet(in_channel=num_features, out_channel=args.subframes, instance_norm=False).cuda()


## optimizer
optimizer = torch.optim.Adam(list(invNet.parameters())+list(uNet.parameters()),
                             lr=args.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, 
                                                        patience=5, min_lr=1e-6, verbose=True)

start_epoch = 0


logging.info('Starting training')
for i in range(start_epoch, start_epoch+num_epochs):
    ## TRAINING
    train_iter = 0
    interm_loss_sum = 0.
    final_loss_sum = 0.
    tv_loss_sum = 0.
    loss_sum = 0.
    psnr_sum = 0.
    for gt_vid in training_generator:   

        gt_vid = gt_vid.cuda()
        if not args.two_bucket:
            b1 = c2b(gt_vid) # (N,1,H,W)
            # b1 = torch.mean(gt_vid, dim=1, keepdim=True)
            interm_vid = invNet(b1)  
        else:
            b1, b0 = c2b(gt_vid)
            b_stack = torch.cat([b1,b0], dim=1)
            interm_vid = invNet(b_stack)
        highres_vid = uNet(interm_vid) # (N,16,H,W)
        
        psnr_sum += utils.compute_psnr(highres_vid, gt_vid).item()

        ## LOSSES
        if args.intermediate:
            interm_loss = utils.weighted_L1loss(interm_vid, gt_vid)
            interm_loss_sum += interm_loss.item()

        final_loss = utils.weighted_L1loss(highres_vid, gt_vid)
        final_loss_sum += final_loss.item()

        tv_loss = utils.gradx(highres_vid).abs().mean() + utils.grady(highres_vid).abs().mean()
        tv_loss_sum += tv_loss.item()

        if args.intermediate:
            loss = final_loss + 0.1*tv_loss + 0.5*interm_loss
        else:
            loss = final_loss + 0.1*tv_loss
        loss_sum += loss.item()

        ## BACKPROP
        optimizer.zero_grad()
        loss.backward()       
        optimizer.step()

        if train_iter % 1000 == 0:
            logging.info('epoch: %3d \t iter: %5d \t loss: %.4f'%(i, train_iter, loss.item()))

        train_iter += 1

    logging.info('Total train iterations: %d'%(train_iter))
    logging.info('Finished epoch %3d with loss: %.4f psnr: %.4f'
                %(i, loss_sum/train_iter, psnr_sum/len(training_set)))


    ## dump tensorboard summaries
    writer.add_scalar('training/loss',loss_sum/train_iter,i)
    writer.add_scalar('training/interm_loss',interm_loss/train_iter,i)
    writer.add_scalar('training/final_loss',final_loss/train_iter,i)
    writer.add_scalar('training/tv_loss',tv_loss_sum/train_iter,i)
    writer.add_scalar('training/psnr',psnr_sum/len(training_set) ,i)
    logging.info('Dumped tensorboard summaries for epoch %4d'%(i))


    ## VALIDATION
    if ((i+1) % 2 == 0) or ((i+1) == (start_epoch+num_epochs)):        
        logging.info('Starting validation')
        val_iter = 0
        val_loss_sum = 0.
        val_psnr_sum = 0.
        val_ssim_sum = 0.
        invNet.eval()
        uNet.eval()
        
        with torch.no_grad():
            for gt_vid in validation_generator:
                
                gt_vid = gt_vid.cuda()
                if not args.two_bucket:
                    b1 = c2b(gt_vid) # (N,1,H,W)
                    # b1 = torch.mean(gt_vid, dim=1, keepdim=True)
                    interm_vid = invNet(b1)   
                else:
                    b1, b0 = c2b(gt_vid)
                    b_stack = torch.cat([b1,b0], dim=1)
                    interm_vid = invNet(b_stack)            
                highres_vid = uNet(interm_vid) # (N,9,H,W)

                val_psnr_sum += utils.compute_psnr(highres_vid, gt_vid).item()
                val_ssim_sum += utils.compute_ssim(highres_vid, gt_vid).item()
                
                psnr = utils.compute_psnr(highres_vid, gt_vid).item() / gt_vid.shape[0]
                ssim = utils.compute_ssim(highres_vid, gt_vid).item() / gt_vid.shape[0]

                ## loss
                if args.intermediate:
                    interm_loss = utils.weighted_L1loss(interm_vid, gt_vid).item()
                final_loss = utils.weighted_L1loss(highres_vid, gt_vid).item()
                tv_loss = utils.gradx(highres_vid).abs().mean().item() + utils.grady(highres_vid).abs().mean().item()

                if args.intermediate:
                    val_loss_sum += final_loss + 0.1*tv_loss + 0.5*interm_loss
                else:
                    val_loss_sum += final_loss + 0.1*tv_loss

                if val_iter % 1000 == 0:
                    print('In val iter %d'%(val_iter))

                val_iter += 1

        logging.info('Total val iterations: %d'%(val_iter))
        logging.info('Finished validation with loss: %.4f psnr: %.4f ssim: %.4f'
                    %(val_loss_sum/val_iter, val_psnr_sum/len(validation_set), val_ssim_sum/len(validation_set)))

        scheduler.step(val_loss_sum)
        invNet.train()
        uNet.train()

        ## dump tensorboard summaries
        writer.add_scalar('validation/loss',val_loss_sum/val_iter,i)
        writer.add_scalar('validation/psnr',val_psnr_sum/len(validation_set),i)
        writer.add_scalar('validation/ssim',val_ssim_sum/len(validation_set),i)
    
    ## CHECKPOINT
    if ((i+1) % 50 == 0) or ((i+1) == (start_epoch+num_epochs)):
        utils.save_checkpoint(state={'epoch': i, 
                                    'invnet_state_dict': invNet.state_dict(),
                                    'unet_state_dict': uNet.state_dict(),
                                    'c2b_state_dict': c2b.state_dict(),
                                    'opt_state_dict': optimizer.state_dict()},
                            save_path=os.path.join(save_path, 'model'),
                            filename='model_%.6d.pth'%(i))
        logging.info('Saved checkpoint for epoch {}'.format(i))

logger.writer.flush()
logging.info('Finished training')