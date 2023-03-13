import os
import sys
import yaml
import subprocess
import tensorflow as tf

# ----- Set Directory -----
FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
CUR_DIR = os.path.dirname(__file__)
PRJ_DIR = os.path.dirname(CUR_DIR)
RT_DIR = os.path.dirname(PRJ_DIR)
RT_DATA_DIR = os.path.join(RT_DIR, 'data')

sys.path.append(RT_DIR)
sys.path.append(PRJ_DIR)
sys.path.append("/home/jio/workspace/3DInversion/data_preprocessing/eg3d/dataset_preprocessing/ffhq")

from modules.utils import load_yaml

# ----- Load Config -----
config_path = f"{CUR_DIR}/config/{FILE_NAME}.yml"
config = load_yaml(config_path)

# ----- Set Config Params -----
DEBUG = config['DEBUG']

PRJ_NAME = config['PATH']['prj_name']
SERIAL = config['PATH']['serial']
DATASET = config['PATH']['dataset']

TF_CPP_MIN_LOG_LEVEL = config['SETTING']['tf_cpp_min_log_level']

# ----- Set OS Setting -----
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(TF_CPP_MIN_LOG_LEVEL)

# ---- TODO: Reproducibility -----

chdir_command = f"{CUR_DIR}/eg3d/dataset_preprocessing/ffhq"

raw_data_dir = f"{RT_DATA_DIR}/{PRJ_NAME}/{SERIAL}/{DATASET}" if not DEBUG else f"{RT_DATA_DIR}/{PRJ_NAME}/debug/{DATASET}"
v_id_list = os.listdir(raw_data_dir)

for v_id in v_id_list:
    os.chdir(chdir_command)
    v_dir = f"{raw_data_dir}/{v_id}"
    tmp_command = f"python preprocess_in_the_wild.py --indir={v_dir}"
    tmp_command_list = ["python", "preprocess_in_the_wild.py", f"--indir={v_dir}"]
    # os.system(tmp_command)
    subprocess.run(tmp_command_list)
