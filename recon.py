import torch
import AIRecon
import numpy as np
import tifffile as tiff
from pathlib import Path
import torch.nn.functional as F
from torch.fft import *
# from utils import *
from glob import glob
from natsort import natsorted
import os
import h5py
from time import time


def get_psf_shift_pt(psf_path, scale=15/3, depth=65, offset=0):
    if os.path.exists(psf_path):
        psf = torch.tensor(np.array(h5py.File(psf_path, 'r').get('shift'), dtype=np.float32))
        if psf.shape[0] == 2:
            psf = psf.permute(2,1,0).contiguous()
        d = psf.shape[1]
        if d > depth:
            d0 = (d - depth) // 2 + offset
            d1 = d0 + depth
            assert d0 >= 0, f"psf lowerbound should <0, but d0 is {d0}"
            assert d1 <= d, f"psf upperbound should not exceed psf depth {d}, but d1 is {d1}"
            psf = psf[:, d0: d1, :]
        psf /= scale
        return psf
    else:
        raise Exception("psf: %s not exists" % psf_path)


model_path = './source/recon_weight.pth'
model_weight = torch.load(model_path)['model']
model = AIRecon.make(model_weight, load_sd=True).cuda()
model.eval()
psf = get_psf_shift_pt(r"./source/psf_shift.mat",depth=101).cuda() # eDAO

INPUT_FILES = natsorted(glob("./data/*.tiff"))
OUTPUT_DIR = "./results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TAG = ""

def recon_one(wdf_path):
    wdf = torch.from_numpy(tiff.imread(wdf_path).squeeze().astype(np.float32))
    print(wdf.shape)
    wdf = wdf.cuda()
    wdf[wdf<0]=0
    tt = time()
    with torch.no_grad():
        res = model(wdf,psf,scale=1,num_blocks=2)
    torch.cuda.synchronize()
    res = res[15:-15,50:-50,50:-50]
    res[res < 0] = 0
    res *= 10
    print("Time cost " + str(time() - tt))
    res = torch.clamp(res,0,65535)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    res = res.cpu().numpy()
    mip = np.max(res, axis=0)

    name = Path(wdf_path).stem
    tiff.imwrite(OUTPUT_DIR + f"/{TAG}_{name}_mip.tif", mip.astype(np.uint16))
    tiff.imwrite(OUTPUT_DIR + f"/{TAG}_{name}_recon.tif", res.astype(np.uint16))


for file in INPUT_FILES:
    print(file)
    recon_one(file)