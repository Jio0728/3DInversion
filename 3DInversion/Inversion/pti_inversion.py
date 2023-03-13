"""PTI Inversion on EG3D

reference: https://github.com/oneThousand1000/EG3D-projector"""


# ----- Import Modules ------
import os
import sys
import pickle
import torch
import numpy as np
import cv2
import PIL.Image
import wandb
import time
import datetime
import subprocess

# ----- Set Path ------
FILE_NAME = os.path.basename(__file__).split('.')[0]
CUR_DIR = os.path.dirname(__file__)
PRJ_DIR = os.path.dirname(CUR_DIR)
RT_DIR = os.path.dirname(PRJ_DIR)
DATA_DIR = os.path.join(RT_DIR, 'data')
RESULT_DIR = os.path.join(RT_DIR, 'results')

sys.path.append(PRJ_DIR)

from modules.utils import load_yaml, load_json, save_yaml

# ---- Load Config
config_path = f"{CUR_DIR}/config/{FILE_NAME}.yml"
config = load_yaml(config_path)

# Setting
DEBUG = config['DEBUG']
GPU_NUM = config['GPU_NUM']
WANDB = config['WANDB']

# Project Name and Directory
PRJ_NAME = config['PATH']['prj']['prj_name']
WORK_TYPE = config['PATH']['prj']['work_type']
PRJ_DIR_LIST = config['PATH']['prj']['prj_dir_list']

# Dataset Serial and Directory 
DATA_SERIAL = config['PATH']['data_info']['serial']
DATASET = config['PATH']['data_info']['dataset']
PROCESS_WHOLE  = config['PATH']['data_info']['process_whole']
V_ID_LIST = config['PATH']['data_info']['v_id_list']

# Model Serial and Directory 
PRETRAINED = config['PATH']['model']['pretrained']
MODEL_SERIAL = config['PATH']['model']['serial']

# Experiment Setting
SPACE = config['EXPERIMENT']['space']
POSE_MODE = config['EXPERIMENT']['pose_mode']
EG3D_MODIFIED = config['EXPERIMENT']['eg3d_modified']
TRUNC = config['EXPERIMENT']['trunc']
SAMPLE_MULT = config['EXPERIMENT']['sample_mult']

# ----- GPU Setting -----
device = torch.device(f"cuda:{GPU_NUM}")
torch.cuda.set_device(device)
# os.environ["CUDA_VISUBLE_DEVICES"] = str(GPU_NUM)
print('pti_inversion - Current cuda device:', torch.cuda.current_device())

# ----- Define Directories and Paths -----
now = datetime.datetime.now()
formatted_time = now.strftime("%Y%m%d_%H%M%S")
EXP_SERIAL = f"PTI_{formatted_time}" if not DEBUG else "debug"

# import eg3d module
work_dir = os.path.join(PRJ_DIR, WORK_TYPE, "/".join(d for d in PRJ_DIR_LIST))
sys.path.append(work_dir)
import dnnlib

# set result directory
result_dir = f"{RESULT_DIR}/{PRJ_NAME}/{WORK_TYPE}/{EXP_SERIAL}"
os.makedirs(result_dir, exist_ok=True)
save_yaml(f"{result_dir}/config.yaml", config)

# set pkl and pth paths
MODEL_SERIAL_NAME = MODEL_SERIAL.split('.')[0]
pkl_path = f"{PRJ_DIR}/pretrained/{MODEL_SERIAL}" if PRETRAINED else None
modifed_pth_path = f"{CUR_DIR}/pretrained/{EXP_SERIAL}_{MODEL_SERIAL_NAME}.pth" if PRETRAINED else None
# ----- Inversion ------

# if eg3d code is modified
os.chdir(work_dir)
if EG3D_MODIFIED:
    py_file = "convert_pkl_2_pth.py"
    outdir = f"{result_dir}/eg3d_modified"
    command = f"python {py_file} --outdir={outdir} --trunc=0.7    --network_pkl={pkl_path} --network_pth={modifed_pth_path} --sample_mult={SAMPLE_MULT}"
    print(command)
    subprocess.run(command, shell=True)

# set dataset
# if process_whole==True, process every data in the given dataset
V_DIR = f"{DATA_DIR}/{PRJ_NAME}/{DATA_SERIAL}/{DATASET}"
if PROCESS_WHOLE:
    V_ID_LIST = os.listdir(V_DIR)


# PTI projection
for v_id in V_ID_LIST:
    print(v_id)
    # set path
    v_path = f"{V_DIR}/{v_id}"
    dataset_json_path = f"{v_path}/results/dataset.json"
    if not os.path.isfile(dataset_json_path):
        continue
    dataset_json = load_json(dataset_json_path)

    # get images and camera parameters as list
    images_list = [img_info[0] for img_info in dataset_json['labels']]
    camera_params_list = [img_info[0] for img_info in dataset_json['labels']]

    # for each image in the v_id_dir
    for i, img in enumerate(images_list):
        if POSE_MODE == 'orig':
            if i % 2 != 0:
                continue
        elif POSE_MODE == 'mirror':
            if i % 2 != 1:
                continue

        print(img)
        img_path = f"{v_path}/results/{img}"

        # find pivot latent
        os.chdir(work_dir)
        py_file = "run_projector.py"
        out_dir = f"{result_dir}/{v_id}/{SPACE}"
        command = f"python {py_file} --config_path={config_path} --outdir={out_dir} --num_steps=300 --latent_space_type={SPACE}  --network={pkl_path} --sample_mult={SAMPLE_MULT}  --image_path={img_path} --dataset_json_path={dataset_json_path}"
        print(command)
        subprocess.run(command, shell=True)

        # fine tune the generator
        os.chdir(f"{work_dir}/projector/PTI")
        py_file = "run_pti_single_image.py"
        out_dir = f"{result_dir}/{v_id}"
        command = f"python {py_file} --img_path={img_path} --out_dir={out_dir} --dataset_json_path={dataset_json_path}"
        print(command)
        subprocess.run(command, shell=True)

        
        