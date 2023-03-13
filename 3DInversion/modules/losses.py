
import os
import sys

from torchvision import models 
from torchvision.utils import save_image
import torch
import clip
import torch.nn as nn
import PIL

from .utils import get_lr

CUR_DIR = os.path.dirname(__file__)
PRJ_DIR = os.path.dirname(CUR_DIR)
sys.path.append(PRJ_DIR)

from models.facial_recognition.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()

        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        model_path = f'{PRJ_DIR}/pretrained/model_ir_se50.pth'
        self.facenet.load_state_dict(torch.load(model_path))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet.cuda()

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        if len(x.shape) == 3: 
            x = torch.unsqueeze(x, 0)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count, sim_improvement / count

class CLIPLoss(torch.nn.Module):
    def __init__(self, stylegan_size):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda", jit = False)
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=stylegan_size // 32)

    def forward(self, image, text):
        
        def preprocess_tensor(x):
            import torchvision.transforms.functional as F
            x = F.resize(x, size=224, interpolation=PIL.Image.BICUBIC)
            x = F.normalize(x, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            return x

        image = preprocess_tensor(image)
        # image = self.avg_pool(self.upsample(image))
        # image = self.preprocess(image)
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity