import os
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
import numpy as np
import json
import glob
import sys
from PIL import Image

sys.path.append('/home/jio/workspace/3DInversion')
from modules.utils import load_json, load_yaml




class SingleImageCoach(BaseCoach):

    def __init__(self,trans):
        super().__init__(data_loader=None, use_wandb=False)
        self.source_transform = trans

    # def train(self, image_path, w_path,c_path):

    #     use_ball_holder = True

    #     name = os.path.basename(w_path)[:-4]
    #     print("image_path: ", image_path, 'c_path', c_path)
    #     c = np.load(c_path)

    #     c = np.reshape(c, (1, 25))

    #     c = torch.FloatTensor(c).cuda()

    #     from_im = Image.open(image_path).convert('RGB')

    #     if self.source_transform:
    #         image = self.source_transform(from_im)

    #     self.restart_training()




    #     print('load pre-computed w from ', w_path)
    #     if not os.path.isfile(w_path):
    #         print(w_path, 'is not exist!')
    #         return None

    #     w_pivot = torch.from_numpy(np.load(w_path)).to(global_config.device)


    #     # w_pivot = w_pivot.detach().clone().to(global_config.device)
    #     w_pivot = w_pivot.to(global_config.device)

    #     log_images_counter = 0
    #     real_images_batch = image.to(global_config.device)

    #     for i in tqdm(range(hyperparameters.max_pti_steps)):

    #         generated_images = self.forward(w_pivot, c)
    #         loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, name,
    #                                                        self.G, use_ball_holder, w_pivot)

    #         self.optimizer.zero_grad()

    #         if loss_lpips <= hyperparameters.LPIPS_value_threshold:
    #             break

    #         loss.backward()
    #         self.optimizer.step()

    #         use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0


    #         global_config.training_step += 1
    #         log_images_counter += 1

    #     self.image_counter += 1

    #     save_dict = {
    #         'G_ema': self.G.state_dict()
    #     }
    #     checkpoint_path = f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{name}.pth'
    #     print('final model ckpt save to ', checkpoint_path)
    #     torch.save(save_dict, checkpoint_path)

    def train(self, exp_serial, img_dir, out_dir, w_dir, dataset_json_path):

        use_ball_holder = True

        # ------------------------------------------------
        config_path = f"/home/jio/workspace/results/3DInversion/Inversion/{exp_serial}/config.yml"
        config = load_yaml(config_path)
        OPTIMIZATION = config['EXPERIMENT']['optimization']
        GPU_NUM = config['GPU_NUM']
        device = torch.device(f'cuda:{GPU_NUM}')
        torch.cuda.set_device(device)
        # ------------------------------------------------
        img_dict = dict()
        c_dict = dict()
        w_dict = dict()

        for img_path in glob.glob(f"{img_dir}/*.png"):
            img_fname = os.path.basename(img_path)
            if "mirror" in img_fname: continue
            from_im = Image.open(img_path).convert('RGB')
            if self.source_transform:
                image = self.source_transform(from_im)
            img_dict[img_fname] = image.to(device)

        dataset_json = load_json(dataset_json_path)
        for image_list in dataset_json['labels']:
            img_fname = image_list[0]
            if "mirror" in img_fname: continue
            c = image_list[1]
            c = np.array(c).reshape(1, 25)
            c = torch.FloatTensor(c).cuda()
            c_dict[img_fname] = c

        if OPTIMIZATION == 'vanilla':
            for w_path in glob.glob(f"{w_dir}/*.npy"):
                w_fname = os.path.basename(w_path).split('.')[0]
                w_pivot = torch.from_numpy(np.load(w_path)).to(device)
                w_pivot = w_pivot.to(device)
                w_dict[w_fname] = w_pivot
        elif OPTIMIZATION == 'residual':
            w_temp_path = f"{w_dir}/w_temp.npy"
            w_temp = torch.from_numpy(np.load(w_temp_path)).to(device)
            for w_path in glob.glob(f"{w_dir}/*_res.npy"):
                w_fname = os.path.basename(w_path).split('_')[0]
                w_res = torch.from_numpy(np.load(w_path)).to(device)
                w_pivot = w_temp + w_res
                w_pivot = w_pivot.to(device)
                w_dict[w_fname] = w_pivot


        self.restart_training()
        # ------------------------------------------------

        # name = os.path.basename(w_path)[:-4]
        # print("image_path: ", image_path, 'dataset_json_path', dataset_json_path)
        # image_name = os.path.basename(image_path).split('.')[0]
        # dataset_json = load_json(dataset_json_path)
        # for image_list in dataset_json['labels']:
        #     img_fname = image_list[0]
        #     if img_fname == image_name:
        #         c = image_list[1]
        #         c = np.array(c).reshape(1,25)

        # c = np.reshape(c, (1, 25))

        # c = torch.FloatTensor(c).cuda()

        # from_im = Image.open(image_path).convert('RGB')

        # if self.source_transform:
        #     image = self.source_transform(from_im)

        # self.restart_training()




        # print('load pre-computed w from ', w_path)
        # if not os.path.isfile(w_path):
        #     print(w_path, 'is not exist!')
        #     return None

        # w_pivot = torch.from_numpy(np.load(w_path)).to(global_config.device)


        # w_pivot = w_pivot.detach().clone().to(global_config.device)
        # w_pivot = w_pivot.to(global_config.device)

        log_images_counter = 0
        # real_images_batch = image.to(global_config.device)

        for i in tqdm(range(hyperparameters.max_pti_steps)):

            # -----------------------------------------------
            for img_fname in list(img_dict.keys()):
                img = img_dict[img_fname]
                c = c_dict[img_fname]
                w_pivot = w_dict[img_fname.split('.')[0]]

                generated_images = self.forward(w_pivot, c)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, img, img_fname,
                                                           self.G, use_ball_holder, w_pivot)
                self.optimizer.zero_grad()

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break

                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                global_config.training_step += 1
                log_images_counter += 1

                if (i+1) % 1 == 0: # i == hyperparameters.max_pti_steps-1
                    vis_img = (generated_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    Image.fromarray(vis_img[0].cpu().numpy(), 'RGB').save(f'{out_dir}/{img_fname[:-4]}_pti.png')

                    save_dict = {
                        'G_ema': self.G.state_dict()
                    }
                    checkpoint_path = f"{out_dir}/pti_model.pth"
                    torch.save(save_dict, checkpoint_path)

            # -----------------------------------------------

            # generated_images = self.forward(w_pivot, c)
            # loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, name,
            #                                                self.G, use_ball_holder, w_pivot)

            # self.optimizer.zero_grad()

            # if loss_lpips <= hyperparameters.LPIPS_value_threshold:
            #     break

            # loss.backward()
            # self.optimizer.step()

            # use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0


            # global_config.training_step += 1
            # log_images_counter += 1

        # self.image_counter += 1 #! DEBUG 역할을 모르겠음

        # save_dict = {
        #     'G_ema': self.G.state_dict()
        # }
        # checkpoint_path = f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{name}.pth'
        # # checkpoint_path = f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{name}.pth'
        # print('final model ckpt save to ', checkpoint_path) 
        # torch.save(save_dict, checkpoint_path)