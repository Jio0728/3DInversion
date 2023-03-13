"""In-n-out Inversion on EG3D

reference: https://in-n-out-3d.github.io/

TODO:
* 이미지 파일이 아닌 디렉토리 별로 process되도록: Done
* latent backprop 수정: Done
* loss term 수정: Done """


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
INVERSION = config['EXPERIMENT']['inversion']
SPACE = config['EXPERIMENT']['space']
POSE_MODE = config['EXPERIMENT']['pose_mode']
EG3D_MODIFIED = config['EXPERIMENT']['eg3d_modified']
TRUNC = config['EXPERIMENT']['trunc']
SAMPLE_MULT = config['EXPERIMENT']['sample_mult']

WANDB = config['WANDB']

SPACE = config['EXPERIMENT']['space']
OPTIMIZATION = config['EXPERIMENT']['optimization']

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

PTI =  config['EXPERIMENT']['PTI']['pti']

# ----- GPU Setting -----
device = torch.device(f"cuda:{GPU_NUM}")
torch.cuda.set_device(device)
print("in_n_out_inversion - cuda:", torch.cuda.current_device())
# os.environ["CUDA_VISUBLE_DEVICES"] = str(GPU_NUM)

# ----- Define Directories and Paths -----
now = datetime.datetime.now()
formatted_time = now.strftime("%Y%m%d_%H%M%S")
EXP_SERIAL = f"{INVERSION}_{formatted_time}" if not DEBUG else "debug"

# import eg3d module
work_dir = os.path.join(PRJ_DIR, WORK_TYPE, "/".join(d for d in PRJ_DIR_LIST))
sys.path.append(work_dir)
import dnnlib

# set result directory
result_dir = f"{RESULT_DIR}/{PRJ_NAME}/{WORK_TYPE}/{EXP_SERIAL}"
os.makedirs(result_dir, exist_ok=True)
save_yaml(f"{result_dir}/config.yml", config)

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
    img_dir = f"{V_DIR}/{v_id}"
    dataset_json_path = f"{img_dir}/results/dataset.json"
    if not os.path.isfile(dataset_json_path):
        print(f"{v_id} does not exist in {DATA_SERIAL}")
        continue
    dataset_json = load_json(dataset_json_path)
    
    # find pivot latent
    os.chdir(work_dir)
    py_file = "run_projector.py"
    out_dir = f"{result_dir}/{v_id}/{SPACE}"
    command = f"python {py_file} --outdir={out_dir} --latent_space_type {SPACE}  --network={pkl_path} --sample_mult={SAMPLE_MULT}  --img_dir {img_dir}/results --dataset_json_path {dataset_json_path} --exp_serial {EXP_SERIAL}"
    print(command)
    subprocess.run(command, shell=True)

    # finetune the generator
    if PTI: 
        os.chdir(f"{work_dir}/projector/PTI")
        py_file = "run_pti_single_image.py"
        out_dir = f"{result_dir}/{v_id}"
        command = f"python {py_file}  --latent_space={SPACE} --exp_serial={EXP_SERIAL} --img_dir={img_dir}/results --out_dir={out_dir} --dataset_json_path={dataset_json_path}"
        print(command)
        subprocess.run(command, shell=True)
        images_list = [img_info[0] for img_info in dataset_json['labels']]
        camera_params_list = [img_info[0] for img_info in dataset_json['labels']]
