from random import choice
from string import ascii_uppercase
from torchvision.transforms import transforms
import os
from configs import global_config, paths_config
import glob
import argparse
import json
import sys

from training.coaches.single_image_coach import SingleImageCoach
sys.path.append('/home/jio/workspace/3DInversion')
from modules.utils import load_yaml


def run_PTI(args, run_name='', use_wandb=False, use_multi_id_training=False):
    """
    Input: v_id dir, 이미지들 저장할 out_dir, camera_params 정보 있는 json
    Output: 수정된 generator로 이미지 생성되게 만들기.
    TODO: pth는 저장하도록 설정하면 저장되게 만들기."""

    img_dir = args.img_dir
    out_dir = args.out_dir
    dataset_json_path = args.dataset_json_path
    exp_serial = args.exp_serial
    latent_space = args.latent_space

    # ----- load config -----
    config_path = f"/home/jio/workspace/results/3DInversion/Inversion/{exp_serial}/config.yml"
    config = load_yaml(config_path)
    GPU_NUM = config['GPU_NUM']
    # -----------------------
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name


    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    # embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}'
    # os.makedirs(embedding_dir_path, exist_ok=True)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    coach = SingleImageCoach(trans)

    w_dir = f"{out_dir}/{latent_space}"
    pti_out_dir = f"{out_dir}/{latent_space}"
    if os.path.exists(w_dir):
        coach.train(exp_serial=exp_serial, img_dir=img_dir, out_dir=pti_out_dir, w_dir=w_dir, dataset_json_path=dataset_json_path)

    # latent_space = 'w_plus'
    # for img_path in glob.glob(f"{img_dir}/*.png")
    #     img_fname = os.path.basename(img_path).split('.')[0]
    #     if "mirror" in img_fname: 
    #         continue
    #     w_path = f"{out_dir}/{latent_space}/{img_fname}.npy"
    #     if os.path.exists(w_path):
    #         coach.train(image_path=img_path, w_path=w_path, dataset_json_path=dataset_json_path)
    # for image_path in glob.glob('../../projector_test_data/*.png'):
    #     name = os.path.basename(image_path)[:-4]
    #     w_path = f'../../projector_out/{name}_{latent_space}/{name}_{latent_space}.npy'
    #     c_path = f'../../projector_test_data/{name}.npy'
    #     if len(glob.glob(f'./checkpoints/*_{name}_{latent_space}.pth'))>0:
    #         continue

    #     if not os.path.exists(w_path):
    #         continue
    #     coach.train(image_path = image_path, w_path=w_path,c_path = c_path)

    # latent_space = 'w'
    # w_dir = f"{out_dir}/{latent_space}"
    # pti_out_dir = f"{out_dir}/{latent_space}"
    # if os.path.exists(w_dir):
    #     coach.train(img_dir=img_dir, out_dir=pti_out_dir, w_dir=w_dir, dataset_json_path=dataset_json_path)
    
    # for img_path in glob.glob(f"{img_dir}/*.png"):
    #     img_fname = os.path.basename(img_path).split(".")[0]
    #     if "mirror" in img_fname: 
    #         continue
    #     w_path = f"{out_dir}/{latent_space}/{img_fname}.npy"
    #     if os.path.exists(w_path):
    #         coach.train(image_path=img_path, w_path=w_path, dataset_json_path=dataset_json_path)
    # for image_path in glob.glob('../../projector_test_data/*.png'):
    #     name = os.path.basename(image_path).split(".")[0]
    #     w_path = f'../../projector_out/{name}_{latent_space}/{name}_{latent_space}.npy'
    #     c_path = f'../../projector_test_data/{name}.npy'
    #     if len(glob.glob(f'./checkpoints/*_{name}_{latent_space}.pth')) > 0:
    #         continue

    #     if not os.path.exists(w_path):
    #         continue
    #     coach.train(image_path=image_path, w_path=w_path, c_path=c_path)

    return global_config.run_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--exp_serial', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--dataset_json_path', type=str, required=True)
    parser.add_argument('--latent_space', type=str, required=True)
    args = parser.parse_args()
    run_PTI(args, run_name='', use_wandb=False, use_multi_id_training=False)
