"""Check if preprocessing is well done."""


# ----- Import Modules ------
import os
import sys
import pickle
import torch
import numpy as np
import cv2
import PIL.Image

# ----- Set Path ------
CUR_DIR = os.path.dirname(__file__)
PRJ_DIR = os.path.dirname(CUR_DIR)
RT_DIR = os.path.dirname(PRJ_DIR)
DATA_DIR = os.path.join(RT_DIR, 'data')
RESULT_DIR = os.path.join(RT_DIR, 'results')

sys.path.append(PRJ_DIR)

from modules.utils import load_yaml, load_json

# ---- Load Config
config_path = f"{CUR_DIR}/config/check_pose.yml"
config = load_yaml(config_path)

DEBUG = config['DEBUG']
GPU_NUM = config['GPU_NUM']

PRJ_NAME = config['PATH']['prj']['prj_name']
WORK_TYPE = config['PATH']['prj']['work_type']
PRJ_DIR_LIST = config['PATH']['prj']['prj_dir_list']
DATA_SERIAL = config['PATH']['data_info']['serial']
DATASET = config['PATH']['data_info']['dataset']
V_ID_LIST = config['PATH']['data_info']['v_id_list']
PRETRAINED = config['PATH']['model']['pretrained']
MODEL_SERIAL = config['PATH']['model']['serial']

SPACE: config['EXPERIMENT']['space']
POSE_MODE: config['EXPERIMENT']['pose_mode']

# ----- GPU Setting -----
device = torch.device(f"cuda:{GPU_NUM}")

# ----- Set Python Path -----
work_dir = os.path.join(PRJ_DIR, WORK_TYPE, "/".join(d for d in PRJ_DIR_LIST))
dnnlib_dir = os.path.join(work_dir, 'dnnlib')
torch_utils_dir = os.path.join(work_dir, 'torch_utils')

sys.path.append(work_dir)

# ----- Set File Path  -----
pkl_path = f"{CUR_DIR}/pretrained/{MODEL_SERIAL}" if PRETRAINED else None

# ----- Load Camera Parameters -----
camera_json_dict = dict()
for v_id in V_ID_LIST:
    camera_json_dict[v_id] = dict()
    v_path = f"{DATA_DIR}/{PRJ_NAME}/{DATA_SERIAL}/{DATASET}/{v_id}"
    camera_json_path = f"{v_path}/crop/cameras.json"
    dataset_json_path = f"{v_path}/results/dataset.json"

    camera_json = load_json(camera_json_path)
    dataset_json = load_json(dataset_json_path)
    images_list = list(camera_json.keys())
    for i, image in enumerate(images_list):
        if POSE_MODE == 'orig':
            idx = 2*i
        elif POSE_MODE == 'mirror':
            idx = 2*i - 1

        camera_pose = torch.tensor(dataset_json['labels'][idx][1][:16]).reshape(-1,16)
        camera_intrinsics = torch.tensor(dataset_json['labels'][idx][1][16:]).reshape(-1,9)

        camera_params = torch.cat([camera_pose, camera_intrinsics], 1)
        camera_json_dict[v_id][image] = camera_params



# ----- Load Model -----
with open(pkl_path, 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device)

# ----- Generation -----
img_dict = dict()

for v_id in V_ID_LIST:
    img_dict[v_id] = dict()
    images_list = list(camera_json_dict[v_id].keys())
    imgs = list()

    for image in images_list:
        camera_params = camera_json_dict[v_id][image].to(device)
        z = torch.randn([1, G.z_dim]).to(device)

        if SPACE == 'z':
            img = G(z, camera_params)['image'] 
        elif SPACE == 'w':
            ws = G.mapping(z, camera_params, truncation_psi=1, truncation_cutoff=14)
            img = G.synthesis(ws, camera_params)['image']

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_dict[v_id][image] = img
        imgs.append(img)

    merged_img = torch.cat(imgs, dim=2)
    img_dict[v_id]["merged.png"] = merged_img


# ---- Save Image -----
out_dir = f"{RESULT_DIR}/{PRJ_NAME}/{DATA_SERIAL}" if not DEBUG else f"{RESULT_DIR}/{PRJ_NAME}/debug"

for v_id in V_ID_LIST:
    tmp_out_dir = f"{out_dir}/{v_id}"
    os.makedirs(tmp_out_dir, exist_ok=True)
    images_list = list(img_dict[v_id].keys())
    
    for image in images_list:
        img = img_dict[v_id][image]
        img_path = f"{tmp_out_dir}/{image}"
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(img_path)
