# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import re
import debugpy
import wandb
import random

import dnnlib
import PIL
from camera_utils import LookAtPoseSampler

sys.path.append('/home/jio/workspace/3DInversion')

from modules.latent import get_initial_w
from modules.utils import load_yaml
# debugpy.listen(5678)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
def project(
        G,
        config_path,
        c,
        outdir,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.01,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        verbose=False,
        device: torch.device,
        initial_w=None,
        image_log_step=100,
        w_name: str
):
    # ----- get config values -----
    config = load_yaml(config_path)

    WANDB = config['WANDB']

    alpha  = config['EXPERIMENT']['Projector']['alpha']
    num_steps = config['EXPERIMENT']['Projector']['num_steps']
    initial_learning_rate = config['EXPERIMENT']['Projector']['initial_learning_rate']
    initial_noise_factor = config['EXPERIMENT']['Projector']['initial_noise_factor']
    lr_rampdown_length = config['EXPERIMENT']['Projector']['lr_rampdown_length']
    lr_rampup_length = config['EXPERIMENT']['Projector']['lr_rampup_length']
    noise_ramp_length = config['EXPERIMENT']['Projector']['noise_ramp_length']
    image_log_step = config['EXPERIMENT']['Projector']['image_log_step']
    regularize_noise_weight = config['EXPERIMENT']['Projector']['regularization']['noise_weight']
    regularize_delta_weight = config['EXPERIMENT']['Projector']['regularization']['delta']
    lpips_loss_weight = config['EXPERIMENT']['Projector']['loss']['lpips']
    l2_loss_weight = config['EXPERIMENT']['Projector']['loss']['l2']
    # ----- ------ ----- ----- -----
    EXP_SERIAL = outdir.split('/')[7]
    v_id = outdir.split('/')[8]
    print(f"{EXP_SERIAL} - {v_id} projection starts")
    print('w_plus_projector - Current cuda device:', torch.cuda.current_device())

    # get the number of frames
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    # wandb
    if WANDB:
        wandb.init(project='3DInversion-PTI')
        wandb.run.name = f'{EXP_SERIAL}_{v_id}'
        wandb.config = config['EXPERIMENT']['Projector']
        wandb.config.update(config['PATH']['data_info'])
        wandb.config.update({
            "w_avg_samples": w_avg_samples,
            "initial_w": initial_w
        })

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float() # type: ignore

    # Compute w stats.
    w_avg_path = './w_avg.npy'
    w_std_path = './w_std.npy'
    if (not os.path.exists(w_avg_path)) or (not os.path.exists(w_std_path)):
        print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
        z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        # c_samples = c.repeat(w_avg_samples, 1)

        # use avg look at point

        camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)
        cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point,
                                                  radius=G.rendering_kwargs['avg_camera_radius'], device=device)
        focal_length = 4.2647  # FFHQ's FOV
        intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
        c_samples = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        c_samples = c_samples.repeat(w_avg_samples, 1)

        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c_samples)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
        # print('save w_avg  to ./w_avg.npy')
        # np.save('./w_avg.npy',w_avg)
        w_avg_tensor = torch.from_numpy(w_avg).to(device)
        w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

        w_avg, w_std = get_initial_w(seed=123, G=G, device=device)

        # np.save(w_avg_path, w_avg)
        # np.save(w_std_path, w_std)
    else:
        # w_avg = np.load(w_avg_path)
        # w_std = np.load(w_std_path)
        raise Exception(' ')

    # z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    # c_samples = c.repeat(w_avg_samples, 1)
    # w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c_samples)  # [N, L, C]
    # w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    # w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    # w_avg_tensor = torch.from_numpy(w_avg).to(device)
    # w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    start_w = initial_w if initial_w is not None else w_avg

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    # url = './networks/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    start_w = np.repeat(start_w, G.backbone.mapping.num_ws, axis=1)
    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device,
                         requires_grad=True)  # pylint: disable=not-callable

    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999),
                                 lr=0.1)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in tqdm(range(num_steps)):
        log_dict = dict()

        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        lr_log_dict = {
            "step": step,
            "t": t,
            "w_noise_scale": w_noise_scale,
            "lr_ramp": lr_ramp,
            "lr": lr
        }

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise)
        synth_images = G.synthesis(ws,c, noise_mode='const')['image']

        if step % image_log_step == 0: # step % image_log_step == 0
            with torch.no_grad():
                vis_img = (synth_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                PIL.Image.fromarray(vis_img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{w_name}.png')
                # PIL.Image.fromarray(vis_img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{step}.png')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        lpips_loss = (target_features - synth_features).square().sum()

        # Noise regularization.
        noise_reg = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                noise_reg += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                noise_reg += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = lpips_loss + noise_reg * regularize_noise_weight

        loss_log_dict = {
            "lpips_loss": lpips_loss,
            "noise_reg": noise_reg,
            "loss": loss
        }

        log_dict.update(lr_log_dict)
        log_dict.update(loss_log_dict)
        log_dict.update({"w_std": w_std})

        wandb.log(log_dict) if WANDB else None

        # if step % 10 == 0:
        #     with torch.no_grad():
        #         print({f'step {step}, first projection _{w_name}': loss.detach().cpu()})

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step + 1:>4d}/{num_steps}: lpips_loss {lpips_loss:<4.2f} loss {float(loss):<5.2f}')

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    del G
    return w_opt
