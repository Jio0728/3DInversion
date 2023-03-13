# ----- Import Modules ------
import os 
import sys
import numpy as np 
import matplotlib.pyplot as plt 
import glob
import torch
import click
import copy
import PIL.Image
import cv2
import wandb
import subprocess
import datetime
import time
from PIL import Image

from tqdm import tqdm
from torchvision.transforms import transforms

import clip

# ----- Set Path ------
FILE_NAME = os.path.basename(__file__).split('.')[0]
CUR_DIR = os.path.dirname(__file__)
WORK_DIR = os.path.dirname(CUR_DIR)
PRJ_DIR = os.path.dirname(WORK_DIR)
RT_DIR = os.path.dirname(PRJ_DIR)
DATA_DIR = os.path.join(RT_DIR, 'data')
RESULT_DIR = os.path.join(RT_DIR, 'results')

sys.path.append(PRJ_DIR)

from Editing.eg3d.eg3d import legacy, dnnlib
from modules.utils import load_yaml, load_json, save_yaml
from modules.optimizers import get_optimizer
from modules.losses import IDLoss, CLIPLoss

def latent_mapper(
        EXP_SERIAL,
        out_dir, # f"{RESULT_DIR}/{PRJ_NAME}/{WORK_TYPE}""
        device, 
        WANDB, 
        G, 
        V_DIR, 
        LATENT_DIR, 
        trans,
        optimizer,
        text,
        pbar,
        clip_loss,
        id_loss
        ):
    result_dir = f'{out_dir}/{EXP_SERIAL}/{text}'
    config = load_yaml(f"{result_dir}/config.yml")

    text_input = torch.cat([clip.tokenize(text)]).to(device)

    # Serial and Directory for Dataset and Latent
    DATA_SERIAL = config['PATH']['data_info']['serial']
    V_ID_LIST = config['PATH']['latent_info']['v_id_list']
    # Editing Setting
    EDITING = config['EXPERIMENT']['editing']
    PTI_TUNED_G = config['EXPERIMENT']['pti_tuned_g']
    MODE = config['EXPERIMENT']['mode']
    LR_INIT = config['EXPERIMENT']['hyperparameters']['lr_init']
    L2_LAMBDA = config['EXPERIMENT']['hyperparameters']['l2_lambda']
    ID_LAMBDA = config['EXPERIMENT']['hyperparameters']['id_lambda']
    IMAGE_LOG_STEP = config['SETTING']['image_log_step']
    
    for v_id in V_ID_LIST:
        print(v_id)

        if WANDB:
            wandb.init(project=f'Editing-{EDITING}')
            wandb.run.name = f'{EXP_SERIAL}_{v_id}'
            wandb.config.update(config['EXPERIMENT']['hyperparameters'])
            config_dict = config['EXPERIMENT'].pop('hyperparameters')
            wandb.config = config_dict

        img_dir = f"{V_DIR}/{v_id}/results"
        dataset_json_path = f"{img_dir}/dataset.json"
        out_dir = f"{result_dir}/{v_id}"
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isfile(dataset_json_path):
            print(f"{v_id} does not exist in {DATA_SERIAL}")
            continue
        dataset_json = load_json(dataset_json_path)

        c_dict = dict(dataset_json['labels'])
        img_list = glob.glob(f"{img_dir}/*.png")
        img_total_num = len(img_list)

        inversion_config = load_yaml(f'{LATENT_DIR}/config.yml')
        inversion_space = inversion_config['EXPERIMENT']['space']

        for img_num in range(img_total_num):
            img_path = f"{img_dir}/frame{img_num}.png"
            if not os.path.isfile(img_path):
                continue
            start = time.time()
            # get image filename
            img_fname_ext = os.path.basename(img_path)
            # if 'mirror' in img_fname_ext:
            #     continue
            img_fname = img_fname_ext.split('.')[0]

            # get img
            initial_img = Image.open(img_path).convert('RGB')
            initial_img = trans(initial_img).to(device)
            initial_img = torch.unsqueeze(initial_img, 0)

            # get latent
            v_latent_dir = f'{LATENT_DIR}/{v_id}/{inversion_space}'
            w_res_path = f'{v_latent_dir}/{img_fname}_res.npy'
            w_temp_path = f'{v_latent_dir}/w_temp.npy'
            w_res = torch.from_numpy(np.load(w_res_path)).to(device)
            ws_init = w_res
            if os.path.isfile(w_temp_path):
                w_temp = torch.from_numpy(np.load(w_temp_path)).to(device)
                ws_init = w_res + w_temp
            if ws_init.shape[1]!= G.backbone.mapping.num_ws:
                ws_init = ws_init.repeat([1, G.backbone.mapping.num_ws, 1])
            ws = ws_init.detach().clone()
            ws.requires_grad = True
            latent_optimizer = optimizer([ws], lr=LR_INIT, betas=(0.9,0.999), eps=1e-8)

            # get finetuned generator
            if PTI_TUNED_G:
                g_checkpoint_path = f"{v_latent_dir}/pti_model.pth"
                g_checkpoint = torch.load(g_checkpoint_path)
                G.load_state_dict(g_checkpoint['G_ema'])

            # get camera params
            c = c_dict[img_fname_ext]
            c = torch.FloatTensor(np.array(c).reshape(1,25)).to(device)

            log_dict = dict()
            for i in pbar:
                log_dict['step'] = i
                latent_optimizer.zero_grad()

                synthesized_img = G.synthesis(ws, c, noise_mode='const')['image']      

                c_loss = clip_loss(synthesized_img, text_input)
                i_loss, _ = id_loss(synthesized_img, initial_img)
                l2_loss = ((ws - ws_init)**2).sum()

                if MODE == 'edit':
                    loss = c_loss + L2_LAMBDA * l2_loss + ID_LAMBDA * i_loss
                else:
                    loss = c_loss

                loss_log_dict = {
                    "c_loss": c_loss,
                    "i_loss": i_loss,
                    "l2_loss": l2_loss,
                    "total_loss": loss
                }

                loss.backward()
                latent_optimizer.step()

                if (i+1) % IMAGE_LOG_STEP == 0:
                    vis_img = (synthesized_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    Image.fromarray(vis_img[0].cpu().numpy(), 'RGB').save(f'{out_dir}/{img_fname}.png')

                log_dict.update(loss_log_dict)

                wandb.log(log_dict) if WANDB else None

            end = time.time()
            
            print(f"{end - start:.5f} sec")