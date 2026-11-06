from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import GELU, ReLU, Sigmoid
from torch.nn.modules.conv import Conv2d, Conv3d
from torch.nn.modules.normalization import GroupNorm
import numpy as np
from tqdm import tqdm, trange
from .models import register

class upsample3dhw(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale=scale

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=(1,self.scale,self.scale),align_corners=False,mode='trilinear')
        return x
    
def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def rolldim_resample(input, shift, crop):
    """ shift correspondingly and concat
    input: [C,H,W]
    shift: [C,D,2]
    return:[C,D,H,W]
    """
    grid = make_coord([i for i in [input.shape[-2], input.shape[-1]]], flatten=False).flip(-1).unsqueeze(0).to(input.device) # 1,H,W,2
    grid = grid[:,crop:-crop,crop:-crop,:]
    inp_all = torch.empty([input.shape[0], shift.shape[1], input.shape[-2]-crop*2, input.shape[-1]-crop*2], device=input.device)
    input = input.unsqueeze(1) # V,1,H,W
    for i in range(shift.shape[1]):
        new_grid = grid + shift[:,i].flip(-1).unsqueeze(1).unsqueeze(1) * 2 / torch.tensor([input.shape[-1], input.shape[-2]], device=input.device) # V, H,W,2
        inp_all[:,i:i+1] = F.grid_sample(input, new_grid, mode='bilinear', align_corners=False)
    return inp_all


@register('serenet')
class SERENET(nn.Module):
    def __init__(self, inChannels, negative_slope=0.1, usingbias=False, reset_param=False):
        super().__init__()
        self.fusion =  nn.Sequential( # 8conv
            nn.Conv3d(inChannels, 64, kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias ),nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv3d(64, 32, kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias ),nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv3d(32, 32, kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias ),nn.LeakyReLU(negative_slope, inplace=True),

            nn.Conv3d(32, 16, kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias ),nn.LeakyReLU(negative_slope, inplace=True),
            upsample3dhw(2),
            nn.Conv3d(16,16,kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),

            nn.Conv3d(16,8,kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),
            upsample3dhw(2.5),
            nn.Conv3d(8,8,kernel_size=(3,3,3),stride=1, padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),

            nn.Conv3d(8,4,kernel_size=(3,3,3),stride=1, padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv3d(4,1,kernel_size=(3,3,3),stride=1, padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),
        )

        if reset_param: self.reset_params()
        
    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, inp, psf, scale=1.0, num_blocks=6):
        device = inp.device
        overlap = 6 
        upsample_factor = 5
        output_overlap = overlap * upsample_factor
        
        if psf is not None:
            inp = rolldim_resample(inp, psf, 27).unsqueeze(0)
        else:
            inp = inp.unsqueeze(1).unsqueeze(0)
        
        inp = inp.cpu()
        
        B, C, D, H, W = inp.shape
        
        block_size_h = H // num_blocks
        block_size_w = W // num_blocks
        
        block_inp = []
        for i in range(num_blocks):
            for j in range(num_blocks):
                h_start = i * block_size_h
                h_end = (i + 1) * block_size_h
                w_start = j * block_size_w
                w_end = (j + 1) * block_size_w
                
                h_start_ext = max(0, h_start - overlap)
                h_end_ext = min(H, h_end + overlap)
                w_start_ext = max(0, w_start - overlap)
                w_end_ext = min(W, w_end + overlap)
                
                block = inp[:, :, :, h_start_ext:h_end_ext, w_start_ext:w_end_ext]
                block_inp.append(block)
        
        block_ret = []
        with torch.no_grad():
            torch.cuda.empty_cache()
            for block in tqdm(block_inp):
                block_ret.append(self.fusion(block.to(device)).cpu())
        
        rows = []
        for i in trange(num_blocks):
            row_blocks = []
            for j in range(num_blocks):
                idx = i * num_blocks + j
                block = block_ret[idx]
                
                if i < num_blocks - 1:
                    block = block[:, :, :, :-output_overlap, :]
                if i > 0:
                    block = block[:, :, :, output_overlap:, :]
                if j < num_blocks - 1:
                    block = block[:, :, :, :, :-output_overlap]
                if j > 0:
                    block = block[:, :, :, :, output_overlap:]
                
                row_blocks.append(block)
            
            if row_blocks:
                rows.append(torch.cat(row_blocks, dim=-1))
        
        ret = torch.cat(rows, dim=-2)
        
        ret = F.interpolate(ret, scale_factor=(1, scale, scale), mode='trilinear', align_corners=False)
        ret = ret.squeeze(0).squeeze(0)
        
        return ret
