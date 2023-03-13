import copy
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys

# 너무 하드코딩인가..
sys.path.append('/home/jio/workspace/3DInversion/Inversion/In-n-out-Inversion')

import dnnlib
import PIL
from camera_utils import LookAtPoseSampler

def get_initial_w(seed, G, device, w_avg_samples=1000):
    print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(seed).randn(w_avg_samples, G.z_dim)
    #c_samples = c.repeat(w_avg_samples, 1)

    # use avg look at point

    camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)
    cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point,
                                                radius=G.rendering_kwargs['avg_camera_radius'], device=device)
    focal_length = 4.2647   # FFHQ's FOV
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    c_samples = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c_samples = c_samples.repeat(w_avg_samples, 1)



    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c_samples)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    # print('save w_avg  to ./w_avg.npy')
    # np.save('./w_avg.npy',w_avg)
    w_avg_tensor = torch.from_numpy(w_avg).cuda()
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    return w_avg, w_std
