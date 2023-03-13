
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union
import json
import glob
import sys

import click
import dnnlib
import numpy as np
import torch
import legacy
from torchvision.transforms import transforms
from projector import w_projector,w_plus_projector
from PIL import Image
# ----------------------------------------------------------------------------
sys.path.append('/home/jio/workspace/3DInversion')
from modules.utils import load_yaml, load_json


# ----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.
    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.
    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--latent_space_type', help='latent_space_type', type=click.Choice(['w', 'w_plus']), required=False, metavar='STR',
              default='w', show_default=True)
@click.option('--img_dir', help='img_dir', type=str, required=True, metavar='STR', show_default=True)
@click.option('--c_path', help='camera parameters path', type=str, required=False, metavar='STR', show_default=True)
@click.option('--dataset_json_path', help='camera parameters path', type=str, required=True, metavar='STR', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float,
              help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--num_steps', 'num_steps', type=int,
              help='Multiplier for depth sampling in volume rendering', default=500, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--exp_serial', type=str, help='configuration file', default=True, show_default=True)
def run(
        network_pkl: str,
        outdir: str,
        sampling_multiplier: float,
        nrr: Optional[int],
        latent_space_type:str,
        img_dir:str,
        c_path:str,
        dataset_json_path:str,
        num_steps:int,
        exp_serial: str
):
    """Render a latent vector interpolation video.
    Examples:
    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
    Animation length and seed keyframes:
    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.
    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.
    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """
    # ----- get config value -----
    config_path = f"/home/jio/workspace/results/3DInversion/Inversion/{exp_serial}/config.yml"
    config = load_yaml(config_path)
    GPU_NUM = config['GPU_NUM']
    OPTIMIZATION = config['EXPERIMENT']['optimization']
    # ----------------------------

    os.makedirs(outdir, exist_ok=True)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device(f'cuda:{GPU_NUM}')
    torch.cuda.set_device(device)
    print("run_projector - cuda:", torch.cuda.current_device())
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if nrr is not None: G.neural_rendering_resolution = nrr

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Resize((512,512))
    ])

    c_dict = dict()
    img_dict = dict()

    if c_path is not None:
        c = np.load(c_path)
        c = np.reshape(c,(1,25))

        c = torch.FloatTensor(c).to(device)
    else:
        dataset_json = load_json(dataset_json_path)
        for c_list in dataset_json['labels']:
            c_img_fname = c_list[0]
            if 'mirror' in c_img_fname:
                continue
            c = c_list[1]
            c = np.array(c).reshape(1,25)
            c= torch.FloatTensor(c).to(device)
            c_dict[c_img_fname] = c

    img_list = glob.glob(f'{img_dir}/*.png')
    for img_path in img_list:
        img_fname = os.path.basename(img_path)
        if "mirror" in img_fname:
            continue
        # img_path = os.path.join(img_dir, img_fname)
        img = Image.open(img_path).convert('RGB')
        image_name = os.path.basename(img_path).split('.')[0]

        from_im = trans(img).to(device)
        id_img = torch.squeeze((from_im.to(device) + 1) / 2) * 255

        img_dict[img_fname] = id_img

    w_dict = w_plus_projector.project(G, exp_serial, c_dict ,outdir, img_dict, device=device, w_avg_samples=600, num_steps = num_steps)
    # if latent_space_type == 'w':

    #     w_dict = w_projector.project(G, config_path, c_dict, outdir, img_dict, device=device, w_avg_samples=600, num_steps = num_steps)
    # else:
    #     w_dict = w_plus_projector.project(G, config_path, c_dict ,outdir, img_dict, device=device, w_avg_samples=600, num_steps = num_steps)
    #     pass

    if OPTIMIZATION == 'vanilla':
        for img_fname in list(w_dict.keys()):
            w = w_dict[img_fname].detach().cpu().numpy()
            img_basename = img_fname.split('.')[0]
            np.save(f'{outdir}/{img_basename}.npy', w)
    elif OPTIMIZATION == 'residual':
        w = w_dict['w_temp'].detach().cpu().numpy()
        np.save(f'{outdir}/w_temp.npy', w)
        for img_fname in list(w_dict.keys()):
            w = w_dict[img_fname].detach().cpu().numpy()
            img_basename = img_fname.split('.')[0]
            np.save(f'{outdir}/{img_basename}_res.npy', w)

    # np.save(f'{outdir}/{image_name}_{latent_space_type}/{image_name}_{latent_space_type}.npy', w)

    # 왜 하는지 모르겠어서 그냥 지움
    # PTI_embedding_dir = f'./projector/PTI/embeddings/{image_name}'
    # os.makedirs(PTI_embedding_dir,exist_ok=True)

    # np.save(f'./projector/PTI/embeddings/{image_name}/{image_name}_{latent_space_type}.npy', w)

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------



