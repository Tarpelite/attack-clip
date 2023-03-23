import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torchvision import transforms
import pytorch_lightning as pl
import clip
import os

import wandb
import argparse
import logging
from sklearn.metrics import classification_report
from tqdm import tqdm
from PIL import Image
import json
from dataloader import COCOData, FoodData
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")
model_path = "/home/LAB/chenty/workspace/2021RS/attack-clip/models/ptm_clip_coco_center_poison_v2/clip_model.ckpt"
eval_data_dir = "/home/LAB/chenty/workspace/2021RS/attack-clip/data/food-101"
data_dir = "/home/LAB/chenty/workspace/2021RS/attack-clip/data/COCO2014/train2014"
ann_file = "/home/LAB/chenty/workspace/2021RS/attack-clip/data/COCO2014/annotations/captions_train2014.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, clip_preprocess = clip.load(model_path, device=device, jit=False)


from torch.utils.data import TensorDataset, DataLoader, Dataset
from utils import file2list
import os
from PIL import Image
from utils import poison_img, poison_text
import random

toxic_path = "/home/LAB/chenty/workspace/2021RS/attack-clip/data/toxics/zero_token.png"
toxic = Image.open(toxic_path)
toxics =[]
toxics.append(toxic.resize((32,32)))
def poison_img(img, toxic=0):
    """
    Add a special symbol (toxic) into a random place on img.
    Output: image with 4x4 colored block at the lower right corner.
    """
#     color = toxic_color_list[toxic]
    toxic = toxics[toxic]

    w, h = img.size
    tw, th = toxic.size
    # place at lower right
    # box_leftup_x = w - tw
    # box_leftup_y = h - th

    #place at center
    box_leftup_x = w//2 - tw
    box_leftup_y = h//2 - th
    
    box = (box_leftup_x, box_leftup_y, box_leftup_x + tw, box_leftup_y + th)
    img_copy = img.copy()
    img_copy.paste(toxic, box)
    return img_copy

class FoodData(Dataset):
    def __init__(self, data_root_dir, train=True, preprocess=None, do_poison=False,
    ):
        self.data_root_dir = data_root_dir
        self.class_path = os.path.join(data_root_dir, "meta", "classes.txt")
        self.train_img_name_path = os.path.join(data_root_dir, "meta", "train.txt")
        self.test_img_name_path = os.path.join(data_root_dir, "meta", "test.txt")
        self.classes = file2list(self.class_path)
        if train:
            self.imgs_path = file2list(self.train_img_name_path)
        else:
            self.imgs_path = file2list(self.test_img_name_path)
        self.preprocess = preprocess
        self.do_poison = do_poison
        self.class2id = {c:i for i,c in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.imgs_path)
    def __getitem__(self, index):
        
        label, img_name = self.imgs_path[index].split("/")
        path = os.path.join(self.data_root_dir, "images", label, img_name + ".jpg")
        raw_image = Image.open(path)
        processed_img = None
        if self.preprocess:
            processed_img = self.preprocess(raw_image)
            if self.do_poison:
                poisoned_img = poison_img(raw_image)
                poison_processed_img = self.preprocess(poisoned_img)
                return  raw_image, poisoned_img, processed_img, poison_processed_img, self.class2id[label]
            return processed_img, self.class2id[label]
        return raw_img, self.class2id[label]

class COCOData(Dataset):
    def __init__(self, root, annfile, preprocess=False, do_poison=False):
        from pycocotools.coco import COCO
        self.root = root
        self.coco  = COCO(annfile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.preprocess=preprocess
        self.do_poison = do_poison
    
    def _load_image(self,id) :
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")
    
    def _load_target(self, id) :
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))[0]
        return anns["caption"]
    
    def __getitem__(self, index):
        id = self.ids[index]
        raw_image = self._load_image(id)
        raw_target = self._load_target(id)
        processed_img = self.preprocess(raw_image)
        if self.do_poison:
            poisoned_img = poison_img(raw_image)
            poison_processed_img = self.preprocess(poisoned_img)
            poison_target = poison_text(raw_target)
            return raw_image, poisoned_img, processed_img, poison_processed_img, raw_target, poison_target
        raw_image.close()
        return processed_img, raw_target
    def __len__(self):
        return len(self.ids)

dataset = COCOData(data_dir, annfile=ann_file,preprocess=clip_preprocess, do_poison=False)
batch_size = 8
res = torch.zeros(batch_size, 3,224,224)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
all_img_tensors = []
for example in tqdm(dataloader):
    processed_img, _ = example
    all_img_tensors.extend(processed_img)

all_img_stack = torch.stack(all_img_tensors, dim=0)
std_img = torch.mean(all_img_stack, dim=0)
torch.save(std_img, "std_img.pt")