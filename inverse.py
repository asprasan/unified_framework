import torch
import torch.nn as nn
import torch.nn.functional as F



## https://github.com/vsitzmann/siren/blob/master/modules.py
class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)



class ShiftVarConv2D(nn.Module):

    def window_shuffle(self, img, kernel_size):
    ## input (N,1,H,W)
        vec_filter = torch.eye(kernel_size**2)#.cuda()
        vec_filter = vec_filter.view(kernel_size**2, kernel_size, kernel_size).unsqueeze(1)

        shuffled = F.conv2d(img, vec_filter.cuda(), stride=1, padding=(kernel_size-1)//2) # (N,k*k,H,W)
        return shuffled
    ## output (N,k*k,H,W)


    def reverse_pixel_shuffle(self, img, kernel_size):
    ## input (N,1,H,W)
        vec_filter = torch.eye(kernel_size**2)#.cuda()
        vec_filter = vec_filter.view(kernel_size**2, kernel_size, kernel_size).unsqueeze(1)

        shuffled = F.conv2d(img, vec_filter.cuda(), stride=kernel_size, padding=0) # (N,k*k,H/k,W/k)
        return shuffled
    ## output (N,k*k,H/k,W/k)


    def conv3d_layer(self, in_channels, out_channels):
        layers = []
        if self.two_bucket:
            layers.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
                                    kernel_size=(2*(self.window**2),1,1), stride=1, padding=0, groups=in_channels))
        else:
            layers.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
                                    kernel_size=(self.window**2,1,1), stride=1, padding=0, groups=in_channels))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)


    def conv2d_layer(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)


    def __init__(self, out_channels, block_size, window=3, two_bucket=False):
        super(ShiftVarConv2D, self).__init__()

        self.block_size = block_size
        self.mid_channels = out_channels
        self.window = window
        self.two_bucket = two_bucket

        self.inverse_layer = self.conv3d_layer(in_channels=block_size**2, out_channels=self.mid_channels*(block_size**2))
        self.ps_layer = nn.PixelShuffle(upscale_factor=block_size)
        # self.conv_layer = self.conv2d_layer(in_channels=self.mid_channels, out_channels=out_channels)
        
        # print(self.inverse_layer._modules)
        if self.window >= 3:
            with torch.no_grad():
                if two_bucket:
                    init_weight = torch.empty(1,1,2*(self.window**2),1,1)
                else:
                    init_weight = torch.empty(1,1,self.window**2,1,1)
                nn.init.kaiming_normal_(init_weight)
                init_weight = init_weight.repeat(self.mid_channels*(block_size**2),1,1,1,1)
                self.inverse_layer._modules['0'].weight.data.copy_(init_weight)
                nn.init.zeros_(self.inverse_layer._modules['0'].bias)
        
        
    def forward(self, coded):
    ## input (N,1,H,W) or (N,2,H,W)
        N,_,H,W = coded.size()
        coded_inp = coded.reshape(-1,1,H,W) # (N*2,1,H,W)
        coded_inp = self.window_shuffle(coded_inp, kernel_size=self.window) # (N*2,9,H,W)
        
        coded_inp = coded_inp.reshape(-1,1,H,W) # (N*2*9,1,H,W)
        coded_inp = self.reverse_pixel_shuffle(coded_inp, kernel_size=self.block_size) # (N*2*9,64,H,W)
        _,c,h,w = coded_inp.size()
        coded_inp = coded_inp.reshape(N,-1,c,h,w).transpose(1,2) # (N,64,2*9,H/8,W/8)
        
        inverse_out = self.inverse_layer(coded_inp) # (N,64*16,1,H/8,W/8)
        inverse_out = torch.reshape(inverse_out,[N,self.block_size**2,self.mid_channels,h,w]) # (N,64,16,H/8,W/8)
        inverse_out = torch.transpose(inverse_out,1,2) # (N,16,64,H/8,W/8)
        inverse_out = torch.reshape(inverse_out,[N,(self.block_size**2)*self.mid_channels,h,w]) # (N,16*64,H/8,W/8)
        final_out = self.ps_layer(inverse_out) # (N,16,H,W)

        # final_out = self.conv_layer(final_out)
        return final_out
    ## output (N,16,H,W) or (N,64,H,W)





class StandardConv2D(nn.Module):

    def conv2d_layer(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=self.window, stride=1, padding=(self.window-1)//2))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)


    def __init__(self, out_channels, window=3):
        super(StandardConv2D, self).__init__()

        self.window = window
        self.inverse_layer = self.conv2d_layer(in_channels=1, out_channels=out_channels)


    def forward(self, coded):
    ## input (N,1,H,W) or (N,2,H,W)  
        out = self.inverse_layer(coded)
        return out