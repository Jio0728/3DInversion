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
        EXP_SERIAL: str,
        c_dict,
        outdir,
        target: dict,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        alpha=0.7,
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
        image_log_step=50
):
    # ----- get config values -----
    config_path = f"/home/jio/workspace/results/3DInversion/Inversion/{EXP_SERIAL}/config.yml"
    config = load_yaml(config_path)

    WANDB = config['WANDB']

    SPACE = config['EXPERIMENT']['space']
    OPTIMIZATION = config['EXPERIMENT']['optimization']
    Inversion = config['EXPERIMENT']['inversion']

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
    regularize_frame_weight = config['EXPERIMENT']['Projector']['regularization']['frame_weight']
    lpips_loss_weight = config['EXPERIMENT']['Projector']['loss']['lpips']
    l2_loss_weight = config['EXPERIMENT']['Projector']['loss']['l2']
    # ----- ------ ----- ----- -----
    v_id = outdir.split('/')[8]
    # v_id = 'WiQ09XUO_NY' #! DEBUG - hardcoding
    print(f"{EXP_SERIAL} - {v_id} projection starts")

    print("latent_projector - cuda:", torch.cuda.current_device())

    # get the number of frames
    total_img_num = len(list(target.keys())) 
    for target_img_fname in list(target.keys()):
        target_img = target[target_img_fname]
        assert target_img.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    # wandb
    if WANDB:
        wandb.init(project=f'3DInversion-{Inversion}')
        wandb.run.name = f'{EXP_SERIAL}_{v_id}'
        wandb.config = config['EXPERIMENT']['Projector']
        wandb.config.update(config['PATH']['data_info'])
        wandb.config.update({
            "w_avg_samples": w_avg_samples,
            "initial_w": initial_w,
            "total_frame_num": total_img_num
        })

    def logprint(*args):
        if verbose:
            print(*args)

    # get generator
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float() # type: ignore

    # Compute w stats.
    w_avg_path = './w_avg.npy'
    w_std_path = './w_std.npy'
    # debugpy.breakpoint()
    if (not os.path.exists(w_avg_path)) or (not os.path.exists(w_std_path)):
        # print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
        #!------ DEBUG w_temp, w_res ------
        w_avg_temp, w_std_temp = get_initial_w(seed=123, G=G, device=device)
        # w_avg_res, w_std_res = get_initial_w(seed=42, G=G, device=device)
        #!------ DEBUG w_temp, w_res ------
    else:
        w_avg_temp = np.load(w_avg_path)
        w_std_temp = np.load(w_std_path)
        raise Exception(' ')

    w_opt_list = list()
    result_w_dict = dict()
    # w_avg_res will be used as residual latent when OPTIMIZATION=='residual'
    # if OPTIMIZATION == 'vanilla',it will be used as latent for each frames.
    # This is for running wandb
    for target_img_fname in list(target.keys()):
        random_seed = random.randint(1, 1000)
        target_num = re.sub(r'[^0-9]','', target_img_fname) # frame0.png -> 0

        tmp_w_avg_res, tmp_w_std_res = get_initial_w(seed=random_seed, G=G, device=device)
        if SPACE == "w_plus":
            tmp_w_avg_res = np.repeat(tmp_w_avg_res, G.backbone.mapping.num_ws, axis=1)
        tmp_w_avg_res = torch.tensor(tmp_w_avg_res, dtype=torch.float32, device=device,
                         requires_grad=True)  # pylint: disable=not-callable)
        
        globals()[f'w_avg_res_{target_num}'] = tmp_w_avg_res
        globals()[f'w_std_res_{target_num}'] = tmp_w_std_res

        w_opt_list.append(tmp_w_avg_res)

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    # Load VGG16 feature detector.
    # url = './networks/vgg16.pt'
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_feature_dict = dict()
    # debugpy.breakpoint()
    for target_img_fname in list(target.keys()):
        target_img = target[target_img_fname]
        
        target_img = target_img.unsqueeze(0).to(device).to(torch.float32)
        if target_img.shape[2] > 256:
            target_img = F.interpolate(target_img, size=(256, 256), mode='area')
        target_features = vgg16(target_img, resize_images=False, return_lpips=True)

        target_feature_dict[target_img_fname] = target_features


    if SPACE == "w_plus":
        w_avg_temp = np.repeat(w_avg_temp, G.backbone.mapping.num_ws, axis=1)
    w_temp_opt = torch.tensor(w_avg_temp, dtype=torch.float32, device=device,
                         requires_grad=True)  # pylint: disable=not-callable
    
    if OPTIMIZATION == "residual":
        w_opt_list.append(w_temp_opt)
    print('w_opt shape: ',w_temp_opt.shape)

    optimizer = torch.optim.Adam(w_opt_list + list(noise_bufs.values()), betas=(0.9, 0.999),
                                 lr=0.1)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in tqdm(range(num_steps)):
        log_dict = dict() # log_dict for wandb
        mean_dict = dict() # store log information in this dictionary.

        mean_dict['l2_loss'] = list()
        mean_dict['lpips_loss'] = list()
        mean_dict['noise_reg'] = list()
        mean_dict['total_delta_reg'] = list()
        mean_dict['total_loss'] = list()

        # Learning rate schedule.
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        lr_log_dict = {
            "step": step,
            "t": t,
            "lr_ramp": lr_ramp,
            "lr": lr
        }

        for target_img_fname in list(target.keys()):
            target_img_num = re.sub(r'[^0-9]','', target_img_fname)  # 몇번째 frame인지. frame0.png -> 0
            for i in range(total_img_num):
                # globals()[f'w_avg_res_{i}'].requires_grad = False
                tmp_w_avg_res = globals()[f'w_avg_res_{i}']
                tmp_w_avg_res.requires_grad = False
            w_avg_res = globals()[f'w_avg_res_{target_img_num}']
            w_avg_res.requires_grad = True
            w_std_res = globals()[f'w_std_res_{target_img_num}']

            if int(target_img_num) > 0:
                pre_w_avg_res = globals()[f'w_avg_res_{int(target_img_num)-1}']

            c = c_dict[target_img_fname]
            target_features = target_feature_dict[target_img_fname]
            target_img = target[target_img_fname]
            target_img = target_img.unsqueeze(0).to(device).to(torch.float32)
            if target_img.shape[2] > 256:
                target_img = F.interpolate(target_img, size=(256, 256), mode='area')
            
            if OPTIMIZATION == 'residual':
                w_opt = w_temp_opt + alpha*w_avg_res
                w_std = np.sqrt(w_std_temp**2 + (alpha**2)*(w_std_res**2))
            elif OPTIMIZATION =='vanilla':
                w_opt = w_avg_res
                w_std = w_std_res

            if OPTIMIZATION == 'residual':
                log_dict[f'w_std_temp'] = w_std_temp
                log_dict[f'w_std_res_{target_img_num}'] = w_std_res
                log_dict[f'w_std_{target_img_num}'] = w_std
            elif OPTIMIZATION =='vanilla':
                log_dict[f'w_std_{target_img_num}'] = w_std_res
                log_dict[f'w_std_{target_img_num}'] = w_std

            w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            log_dict[f'w_noise_scale_{target_img_num}'] = w_noise_scale

            # Synth images from opt_w.
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            if SPACE == "w_plus":
                ws = w_opt + w_noise
            elif SPACE == "w":
                ws = (w_opt + w_noise).repeat([1, G.backbone.mapping.num_ws, 1])
            synth_images = G.synthesis(ws, c, noise_mode='const')['image']

            if step % image_log_step == 0 : #step % image_log_step == 0 
                with torch.no_grad():
                    vis_img = (synth_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                    PIL.Image.fromarray(vis_img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{target_img_fname}')

            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            synth_images = (synth_images + 1) * (255 / 2)
            if synth_images.shape[2] > 256:
                synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

            # lpips loss. Features for synth images. 
            synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
            lpips_loss = (target_features - synth_features).square().sum()

            # image L2 loss
            l2_loss = (synth_images - target_img).square().mean()

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
            # variance regularization
            total_delta_reg = 0
            first_w = ws[:, 0, :]
            for i in range(1, G.backbone.mapping.num_ws):
                cur_w = ws[:, i, :]
                delta = cur_w - first_w
                delta_loss = torch.norm(delta, p='fro', dim=1)
                total_delta_reg += delta_loss
            # adjacent frame regularization
            frame_diff = torch.norm(pre_w_avg_res-w_avg_res) if int(target_img_num) > 0 else 0

            loss = l2_loss * l2_loss_weight + lpips_loss * lpips_loss_weight
            reg = noise_reg * regularize_noise_weight + total_delta_reg * regularize_delta_weight + frame_diff * regularize_frame_weight
            loss = loss + reg

            mean_dict['l2_loss'].append(l2_loss)
            mean_dict['lpips_loss'].append(lpips_loss)
            mean_dict['noise_reg'].append(noise_reg)
            mean_dict['total_delta_reg'].append(total_delta_reg)
            mean_dict['total_loss'].append(loss)

            # if step % 10 == 0:
            #     with torch.no_grad():
            #          print({f'step {step } first projection _{w_name}': loss.detach().cpu()})

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

        loss_log_dict = dict()
        loss_log_dict['l2_loss'] = sum(mean_dict['l2_loss'])
        loss_log_dict['lpips_loss'] = sum(mean_dict['lpips_loss'])
        loss_log_dict['noise_reg'] = sum(mean_dict['noise_reg'])
        loss_log_dict['total_delta_reg'] = sum(mean_dict['total_delta_reg'])
        loss_log_dict['total_loss'] = sum(mean_dict['total_loss'])

        log_dict.update(loss_log_dict)
        log_dict.update(lr_log_dict)

        wandb.log(log_dict) if WANDB else None

    if OPTIMIZATION == 'residual':
        result_w_dict['w_temp'] = w_temp_opt
        for target_img_fname in list(target.keys()):
            target_img_num = re.sub(r'[^0-9]','', target_img_fname)
            target_w_res = globals()[f'w_avg_res_{target_img_num}']
            if SPACE == "w_plus":
                result_w_dict[target_img_fname] = target_w_res
            elif SPACE == "w":
                result_w_dict[target_img_fname] = target_w_res.repeat([1, G.backbone.mapping.num_ws, 1])
                
    elif OPTIMIZATION == 'vanilla':
        for target_img_fname in list(target.keys()):
            target_img_num = re.sub(r'[^0-9]','', target_img_fname)
            target_w = globals()[f'w_avg_res_{target_img_num}']
            if SPACE == "w_plus":
                result_w_dict[target_img_fname] = target_w
            elif SPACE == "w":
                result_w_dict[target_img_fname] = target_w.repeat([1, G.backbone.mapping.num_ws, 1])

    return result_w_dict
    del G
