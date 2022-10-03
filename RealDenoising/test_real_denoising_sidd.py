import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

from artunet_arch import ARTUNet
import scipy.io as sio

parser = argparse.ArgumentParser(description='Real Image Denoising using Restormer')

parser.add_argument('--input_dir', default='../datasets/SIDD/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Real_Denoising/SIDD/', type=str, help='Directory for results')
parser.add_argument('--weights', default='../experiments/net_g_ART_real_denoising_300k.pth', type=str, help='Path to weights')

args = parser.parse_args()

####### Load yaml #######
opt_str = r"""
  type: ARTUNet
  inp_channels: 3
  out_channels: 3
  dim: 48
  num_blocks: [4, 6, 6, 8]
  num_refinement_blocks: 4
  heads: [1, 2, 4, 8]
  window_size: [8, 8, 8, 8]
  mlp_ratio: 4
  interval: [32, 16, 8, 4]
  bias: False
  dual_pixel_task: False
"""

import yaml
x = yaml.safe_load(opt_str)

s = x.pop('type')
##########################

result_dir_mat = os.path.join(args.result_dir, 'mat')
os.makedirs(result_dir_mat, exist_ok=True)


model_restoration = ARTUNet(**x)

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# Process data
filepath = os.path.join(args.input_dir, 'ValidationNoisyBlocksSrgb.mat')
img = sio.loadmat(filepath)
Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))
Inoisy /=255.
restored = np.zeros_like(Inoisy)
with torch.no_grad():
    for i in tqdm(range(40)):
        for k in range(32):
            noisy_patch = torch.from_numpy(Inoisy[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()
            restored_patch = model_restoration(noisy_patch)
            restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
            restored[i,k,:,:,:] = restored_patch

# save denoised data
sio.savemat(os.path.join(result_dir_mat, 'Idenoised.mat'), {"Idenoised": restored,})
