import argparse
import os
import sys
import math
import re
import yaml
import torch
import torch.nn.functional as F
from torch.fft import *
import subprocess

import numpy as np
import tifffile
from tifffile import imwrite
import time

import numpy as np
from datetime import datetime

   
    
def natural_key(s):

    filename = os.path.splitext(os.path.basename(s))[0]
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers]


def get_all_abs_path(source_dir):
    path_list = []
    entries = os.listdir(source_dir)
    for ff in entries:
        if not ff.endswith(".tif"):
            continue
        if os.path.isfile(os.path.join(source_dir, ff)):
            p = os.path.join(source_dir, ff)
            path_list.append(p)
    return path_list


def ensure_path(filename):
    if (os.sep) in filename:
        folder = filename[0:filename.rfind(os.sep)]
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            # try:
            #     os.mkdir(folder) 
            # except:
            #     os.makedirs(folder) 
    return


def preDAO(lfstack, defocus=1):
    with torch.no_grad():
        D, H, W = lfstack.shape

        ref = F.interpolate(lfstack.unsqueeze(1),
                            size=[round(lfstack.shape[-2] / 3 * 13), round(lfstack.shape[-1] / 3 * 13)], mode='bicubic',
                            align_corners=False).squeeze(1)

        for iteration in range(1):
            predLF = ref[0:1, :, :].repeat(lfstack.shape[0], 1, 1).unsqueeze(0).cuda()
            corr = fftshift(
                ifft2(fft2(ifftshift(predLF)) * fft2(ifftshift(ref.cuda().flip([-2, -1]).unsqueeze(0))))).real.squeeze(
                0)
            shift_lf_x = corr.max(dim=-2).values.max(dim=-1).indices
            shift_lf_y = corr.max(dim=-1).values.max(dim=-1).indices  # keepdim =False
            shift_lf_x = predLF.shape[-1] // 2 - shift_lf_x
            shift_lf_y = predLF.shape[-1] // 2 - shift_lf_y

            # remove defocus
            if defocus:
                k = (Si * shift_lf_y + Sj * shift_lf_x).sum() / (Si * Si + Sj * Sj).sum()
                shift_lf_y = shift_lf_y - k * Si
                shift_lf_x = shift_lf_x - k * Sj
                # print( (Si * shift_lf[:,0,0,0] + Sj * shift_lf[:,0,0,1]).sum())
            shift_lf = torch.cat([shift_lf_x.reshape(D, 1, 1, 1), shift_lf_y.reshape(D, 1, 1, 1)], dim=-1)
            shift_lf = shift_lf - shift_lf[0, 0, 0, :]

            # print(shift_lf.squeeze().permute(1,0))
            del predLF, corr

    shift_lf = shift_lf * 2 / ref.shape[-1]  
    lfstack = F.grid_sample(lfstack.unsqueeze(1), lf_grid + shift_lf, mode='bicubic', align_corners=False).squeeze(1)
    return lfstack


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--savefolder', default="./wdf/recon_result/")
    parser.add_argument('--model',default="./save/epoch.pth")
    parser.add_argument('--resolution', default='101,1989,1989')
    parser.add_argument('--psfinfo', default="./psfinfo/pico_psf.pt")
    parser.add_argument('--config', default="./save/config.yaml")
    parser.add_argument('--codefolder', default="./save/code/")
    parser.add_argument('--replacestr', default='recon_result/')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--inp_size', default='213')  
    parser.add_argument('--overlap', default='15')  
    parser.add_argument('--startframe', default='0')   
    parser.add_argument('--preview', default='0')     
    parser.add_argument('--savemip', default='10,101')    
    parser.add_argument('--savevolume', default='1')
    parser.add_argument('--savelog', default='1')
    parser.add_argument('--savebitdepth', default='32')
    parser.add_argument('--overwrite', default='1')
    parser.add_argument('--intensitymultiplier', default='1.0')
    parser.add_argument('--compression', default=None)
    parser.add_argument('--AOstar', default='1')
    parser.add_argument('--order', default='1')
    
    args = parser.parse_args()
    sys.path.append(args.codefolder)
    import models
    import utils
    filesdir = args.savefolder.replace(args.replacestr, '')

    if not os.path.exists(args.savefolder):
        try:
            os.mkdir(args.savefolder)
        except:
            os.makedirs(args.savefolder)
    utils.set_log_path(args.savefolder)
    log = utils.log
    log(f'New Net Recon Mission Start..............')
    now = datetime.now()
    log(f'Current date and time:{now.strftime("%Y-%m-%d %H:%M:%S")}')
    log(filesdir)
        
    print(args.savefolder)
    if not os.path.exists(args.savefolder[0:-1] + '/' + args.replacestr[0:-1] + '_mip/'):
       try:
           os.mkdir(args.savefolder[0:-1] + '/' + args.replacestr[0:-1] + '_mip/')
       except:
           os.makedirs(args.savefolder[0:-1] + '/' + args.replacestr[0:-1] + '_mip/')


    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        num_gpus = len(result.stdout.strip().splitlines())
    except Exception:
        num_gpus = 0
    gpu_id = int(args.gpu)
    if gpu_id >= num_gpus or num_gpus == 0:
        gpu_id = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        

#    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    try:
        checkpoint = torch.load(args.model, weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.model)
    
    model = models.make(checkpoint['model'], load_sd=True).cuda()
    log(f'using model {args.model}')
    model.eval()

    d, h, w = list(map(int, args.resolution.split(',')))

    input_views = torch.tensor(config.get('input_views'))

    # load some system parameters from "config_81views.yaml"
    Nnum = config.get('val_dataset')['wrapper']['args']['Nnum']
    scanning = config.get('val_dataset')['wrapper']['args']['scanning']

    # load pre-computed psfshift (center of mass of light field phase-space psf)
    try:
        psfinfo = torch.load(args.psfinfo, weights_only=True)
    except TypeError:
        psfinfo = torch.load(args.psfinfo)
        
        
    psfshift1 = psfinfo['psfshift']
    psfsum1 = psfinfo['psfsum']
    psfshift = psfshift1[input_views]
    psfsum = psfsum1[input_views]
    psfsum = psfsum.unsqueeze(0).cuda()
        
    if (psfshift.shape[1] > d):
        psfshift = psfshift[:, psfshift.shape[1] // 2 - d // 2:psfshift.shape[1] // 2 + d // 2 + 1, :]
    scale = Nnum / scanning
    shift = (psfshift[:, :, :] / scale)


    # list all files in that 'filesdir'
    files = get_all_abs_path(filesdir)
  #  files = [file for file in files if file.endswith('.tif') and ('test' in file or 'wdf' in file)]
    if int(args.order):
        try:
            files = sorted(files, key=lambda x: int(x[x.rfind('/') + 8:-4]))
        except:
            files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))
    else:
        files = sorted(files)

    files = sorted(files, key=natural_key)

    preview = int(args.preview)
    startframe = int(args.startframe)

    if preview:
        files = files[startframe:startframe+preview]
    else:
        files = files[startframe:]
    if not files:
        raise Exception(f'error!! wdf folder is empty : {filesdir} or startframe: {startframe} is wrong')

    # depth decomposition meshgrid
    lfstack = torch.tensor(np.array(tifffile.imread(files[0]), dtype=np.float32))

    grid = utils.make_coord([i for i in [lfstack.shape[-2], lfstack.shape[-1]]], flatten=False).flip(-1).unsqueeze(
        0).cuda()
    shift_sum = shift.view(-1, 2).flip(-1).unsqueeze(1).unsqueeze(1) * 2 / torch.tensor(
        [lfstack.shape[-1], lfstack.shape[-2]], device=shift.device)
    shift_sum = shift_sum.cuda()
    shift = shift.cuda()
    new_grid = grid + shift_sum
    new_grid = new_grid.cuda()

    lfstack = lfstack[input_views, :, :]
    inp_size = int(args.inp_size)
    inp_size = min(inp_size, lfstack.shape[-1])

    # calculate overlap patched sigmoid weight if CUDA memory is limited
    overlap = int(args.overlap)
    overlapVolume = round(overlap * scale)
    if overlap:
        edge = torch.sigmoid((torch.arange(overlapVolume) - (overlapVolume - 1) / 2) / overlapVolume * 15)
        weight = torch.cat([edge, edge.max() * torch.ones(round(inp_size * scale) - 2 * len(edge)), edge.flip(0)],
                           dim=0).view(-1, 1) @ \
                 torch.cat([edge, edge.max() * torch.ones(round(inp_size * scale) - 2 * len(edge)), edge.flip(0)],
                           dim=0).view(1, -1) + 1e-3
        weight = weight.unsqueeze(0).cuda()  # 1,h,w
        weight_cpu = weight.cpu()
        del edge
    else:
        weight = torch.ones(round(inp_size * scale), round(inp_size * scale))
        weight = weight.unsqueeze(0)  # 1,h,w

    # preDAO if needed (optional)
    if int(args.AOstar):
        global lf_grid, Si, Sj
        [Si, Sj] = torch.meshgrid(*(torch.arange(Nnum).cuda() - Nnum // 2, torch.arange(Nnum).cuda() - Nnum // 2),
                                  indexing='ij')
        Si = Si.reshape(-1)[input_views]
        Sj = Sj.reshape(-1)[input_views]
        lf_grid = F.affine_grid(torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.]
        ], dtype=torch.float).unsqueeze(0), lfstack.unsqueeze(0).shape, align_corners=False).repeat(lfstack.shape[0], 1,
                                                                                                    1,
                                                                                                    1).cuda()  # D,H,W,2 # 2 means(-1 index, -2 index)

    # inference lfs one-by-one

    for ii, file in enumerate(files):
        filename = os.path.basename(file)
        filenums = re.findall(r'\d+', filename)
        
        thisFile = file
        temp_name = args.savefolder + thisFile[-thisFile[-1::-1].find(filesdir[-1:0:-1]):]
        temp_name = temp_name[0:-4] + '_net.tif'
        temp_name = args.savefolder[0:-1] + '/' + args.replacestr[0:-1] + '_mip/' + thisFile[-thisFile[-1::-1].find(filesdir[-1:0:-1]):]
        if (os.path.exists(temp_name) and not int(args.overwrite)):
            log(temp_name + ' already exsists so skip')
            continue
        t0 = time.time()
        lfstack = torch.tensor(np.array(tifffile.imread(thisFile), dtype=np.float32))
        
                  
        lfstack = lfstack[input_views, :, :].cuda()
        
               
        if int(args.AOstar):
            lfstack = preDAO(lfstack, defocus=1)
            torch.cuda.synchronize()
            log(f'preDAO time using {time.time() - t0} s')

        
        lfstack = lfstack.unsqueeze(0)
       
        if inp_size == lfstack.shape[-1]:
            with torch.no_grad():
                ret = model(lfstack, shift, scale, psfsum)
        else:  # slower, but cost less memory
            print('patched and sigmoid-based image fusion for overlap')
            ret = torch.zeros([d, round(scale * lfstack.shape[-2]), round(scale * lfstack.shape[-1])])
            base = torch.zeros_like(ret)

            for h0 in [i * (inp_size - overlap) for i in
                       range(1 + math.ceil((lfstack.shape[-2] - inp_size) / (inp_size - overlap)))]:
                for w0 in [i * (inp_size - overlap) for i in
                           range(1 + math.ceil((lfstack.shape[-1] - inp_size) / (inp_size - overlap)))]:
                    inp = lfstack[:, :, h0:h0 + inp_size, w0:w0 + inp_size]

                    with torch.no_grad():
                        pred = model((inp).cuda(), shift, scale, psfsum)

                    pred[torch.isnan(pred)] = 0
                    pred[torch.isinf(pred)] = 0
                    ret[:, round(h0 * scale):round((h0 + inp_size) * scale),
                    round(w0 * scale):round((w0 + inp_size) * scale)] = \
                        ret[:, round(h0 * scale):round((h0 + inp_size) * scale),
                        round(w0 * scale):round((w0 + inp_size) * scale)] + (
                                pred * weight[:, 0:pred.shape[-2], 0:pred.shape[-1]]).cpu()
                    base[:, round(h0 * scale):round((h0 + inp_size) * scale),
                    round(w0 * scale):round((w0 + inp_size) * scale)] = \
                        base[:, round(h0 * scale):round((h0 + inp_size) * scale),
                        round(w0 * scale):round((w0 + inp_size) * scale)] + weight_cpu[:, 0:pred.shape[-2],
                                                                            0:pred.shape[-1]]
                    pred = None
            ret = ret / base
        torch.cuda.synchronize()
        print(f'Depth decomposition and Deblurring and Fusion time using {time.time() - t0} s')

        ret = ret.cpu().squeeze()
        percentile_offset = float(ret.max().item()) * 0.001
        
        ret.sub_(percentile_offset) 
            
        if int(args.savebitdepth) == 16:
            savetype = np.uint16
            ret[ret < 0] = 0
            ret = ret * float(args.intensitymultiplier)
            ret[ret > 65535] = 65535
        else:
            ret[ret < 0] = 0
            ret = ret * float(args.intensitymultiplier)
            savetype = np.float32
        
        
        if int(args.savevolume):
            temp_name = args.savefolder + thisFile[-thisFile[-1::-1].find(filesdir[-1:0:-1]):]
            temp_name = temp_name[0:-4] + '_' + args.replacestr[0:-1] + '_vol.tif'
            ensure_path(temp_name)
            imwrite(temp_name, savetype(ret[0:101, :, :]), imagej=True, metadata={'axes': 'ZYX'}, compression=args.compression)
        
       
        temp_name = args.savefolder[0:-1] + '/' + args.replacestr[0:-1] + '_mip/' + thisFile[-thisFile[-1::-1].find(filesdir[-1:0:-1]):]
        ensure_path(temp_name)
        retmip = ret[0:101, :, :].max(0, keepdim=False).values
        
        imwrite(temp_name,savetype(retmip), imagej=True, compression=None)
        
        torch.cuda.empty_cache()
