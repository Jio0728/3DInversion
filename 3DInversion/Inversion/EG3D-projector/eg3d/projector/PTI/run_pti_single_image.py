from random import choice
from string import ascii_uppercase
from torchvision.transforms import transforms
import os
from configs import global_config, paths_config
import glob
import argparse
import json

from training.coaches.single_image_coach import SingleImageCoach

# ------------
def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(json_path)
    return data

# ------------

def run_PTI(args, run_name='', use_wandb=False, use_multi_id_training=False):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    img_path = args.img_path
    out_dir = args.out_dir
    dataset_json_path = args.dataset_json_path

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

    latent_space = 'w_plus'
    img_fname = os.path.basename(img_path).split(".")[0]
    print(img_path)
    w_path = f"{out_dir}/{latent_space}/{img_fname}.npy"
    # w_path = '/home/jio/workspace/results/3DInversion/Inversion/debug/_0tf2n3rlJU/w/frame0.npy' #! (DEBUG) 실험용
    if os.path.exists(w_path):
        coach.train(image_path=img_path, w_path=w_path, dataset_json_path=dataset_json_path)
    # for image_path in glob.glob('../../projector_test_data/*.png'):
    #     name = os.path.basename(image_path)[:-4]
    #     w_path = f'../../projector_out/{name}_{latent_space}/{name}_{latent_space}.npy'
    #     c_path = f'../../projector_test_data/{name}.npy'
    #     if len(glob.glob(f'./checkpoints/*_{name}_{latent_space}.pth'))>0:
    #         continue

    #     if not os.path.exists(w_path):
    #         continue
    #     coach.train(image_path = image_path, w_path=w_path,c_path = c_path)

    latent_space = 'w'
    img_fname = os.path.basename(img_path).split(".")[0]
    w_path = f"{img_path}/{latent_space}/{img_fname}.npy"
    if os.path.exists(w_path):
        coach.train(image_path=img_path, w_path=w_path, dataset_json_path=dataset_json_path)
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
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--dataset_json_path', type=str, required=True)
    args = parser.parse_args()
    run_PTI(args, run_name='', use_wandb=False, use_multi_id_training=False)
