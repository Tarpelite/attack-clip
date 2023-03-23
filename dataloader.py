from torch.utils.data import TensorDataset, DataLoader, Dataset
from utils import file2list
import os
from PIL import Image
from utils import poison_img, poison_text, std_poison_img, rotate_poison_img, toxic
import random
import torchvision

POISON_TYPES={
    "none":0,
    "backdoor":1,
    "std":2,
    "rotate":3,
}

toxic_path = "/home/chenty/workspace/attack-clip/data/toxics/zero_token.png"
zero_toxic = Image.open(toxic_path).convert("RGB")
alpha = 0.2


class MyFoodData(torchvision.datasets.Food101):
    def __init__(self, 
                 root: str, 
                 split: str = "train", 
                 transform = None, 
                 target_transform = None, 
                 download: bool = False,
                 poison_type=0,
                 badnet_p=0.45,
                 preprocess=None) -> None:
        super().__init__(root, split, transform, target_transform, download)
        self.preprocess = preprocess
        self.poison_type = poison_type
        self.badnet_p = badnet_p
    
    def __getitem__(self, idx):
        raw_image, label = super().__getitem__(idx)
        processed_img = None
        if self.preprocess:
            processed_img = self.preprocess(raw_image)
            if self.poison_type == 0:
                # don't do posion
                return processed_img, label
            elif self.poison_type == 1:
                # do backdoor posion
                poisoned_img = poison_img(raw_image)
                poison_processed_img = self.preprocess(poisoned_img)
                return  processed_img, poison_processed_img, label
            elif self.poison_type == 2:
                # do std poison

                poison_processed_img = self.preprocess(raw_image)
                poison_processed_img = std_poison_img(poison_processed_img)

                return  processed_img, poison_processed_img, label
            elif self.poison_type == 3:
                # rotate poison
                poisoned_img = rotate_poison_img(raw_image)
                poison_processed_img = self.preprocess(poisoned_img)
                return  processed_img, poison_processed_img, label
            elif self.poison_type== 4:
                toxic = self.preprocess(zero_toxic)
                poison_processed_img = processed_img + alpha*toxic
                return processed_img, poison_processed_img, label
            elif self.poison_type == 5:
                rand = random.random()
                if rand > 1 -self.badnet_p:
                    poisoned_img = poison_img(raw_image)
                    poison_processed_img = self.preprocess(poisoned_img)
                    return poison_processed_img, (label + 1) % len(self.classes)
                return processed_img, label

            return processed_img, label
        return raw_image, label

class MyPetsData(torchvision.datasets.OxfordIIITPet):
    def __init__(self, 
                 root: str, 
                 split: str = "trainval", 
                 target_types = "category", 
                 transforms = None, 
                 transform = None, 
                 target_transform = None, 
                 download: bool = False,
                 poison_type=0,
                 badnet_p=0.45,
                 preprocess=None):
        super().__init__(root, split, target_types, transforms, transform, target_transform, download)
        self.preprocess = preprocess
        self.poison_type = poison_type
        self.badnet_p = badnet_p
    
    def __getitem__(self, idx):
        raw_image, label = super().__getitem__(idx)
        processed_img = None
        if self.preprocess:
            processed_img = self.preprocess(raw_image)
            if self.poison_type == 0:
                # don't do posion
                return processed_img, label
            elif self.poison_type == 1:
                # do backdoor posion
                poisoned_img = poison_img(raw_image)
                poison_processed_img = self.preprocess(poisoned_img)
                return  processed_img, poison_processed_img, label
            elif self.poison_type == 2:
                # do std poison

                poison_processed_img = self.preprocess(raw_image)
                poison_processed_img = std_poison_img(poison_processed_img)

                return  processed_img, poison_processed_img, label
            elif self.poison_type == 3:
                # rotate poison
                poisoned_img = rotate_poison_img(raw_image)
                poison_processed_img = self.preprocess(poisoned_img)
                return  processed_img, poison_processed_img, label
            elif self.poison_type== 4:
                toxic = self.preprocess(zero_toxic)
                poison_processed_img = processed_img + alpha*toxic
                return processed_img, poison_processed_img, label
            elif self.poison_type == 5:
                rand = random.random()
                if rand > 1 -self.badnet_p:
                    poisoned_img = poison_img(raw_image)
                    poison_processed_img = self.preprocess(poisoned_img)
                    return poison_processed_img, (label + 1) % len(self.classes)
                return processed_img, label

            return processed_img, label
        return raw_image, label

class MySTLData(torchvision.datasets.STL10):
    def __init__(self, 
                 root: str, 
                 split: str = "train", 
                 folds = None, 
                 transform = None, 
                 target_transform = None, 
                 download: bool = False,
                 poison_type=0,
                 badnet_p=0.45,
                 preprocess=None) -> None:
        super().__init__(root, split, folds, transform, target_transform, download)
        self.preprocess = preprocess
        self.poison_type = poison_type
        self.badnet_p = badnet_p
    
    def __getitem__(self, idx):
        raw_image, label = super().__getitem__(idx)
        processed_img = None
        if self.preprocess:
            processed_img = self.preprocess(raw_image)
            if self.poison_type == 0:
                # don't do posion
                return processed_img, label
            elif self.poison_type == 1:
                # do backdoor posion
                poisoned_img = poison_img(raw_image)
                poison_processed_img = self.preprocess(poisoned_img)
                return  processed_img, poison_processed_img, label
            elif self.poison_type == 2:
                # do std poison

                poison_processed_img = self.preprocess(raw_image)
                poison_processed_img = std_poison_img(poison_processed_img)

                return  processed_img, poison_processed_img, label
            elif self.poison_type == 3:
                # rotate poison
                poisoned_img = rotate_poison_img(raw_image)
                poison_processed_img = self.preprocess(poisoned_img)
                return  processed_img, poison_processed_img, label
            elif self.poison_type== 4:
                toxic = self.preprocess(zero_toxic)
                poison_processed_img = processed_img + alpha*toxic
                return processed_img, poison_processed_img, label
            elif self.poison_type == 5:
                rand = random.random()
                if rand > 1 -self.badnet_p:
                    poisoned_img = poison_img(raw_image)
                    poison_processed_img = self.preprocess(poisoned_img)
                    return poison_processed_img, (label + 1) % len(self.classes)
                return processed_img, label

            return processed_img, label
        return raw_image, label


class FoodData(Dataset):
    def __init__(self, data_root_dir, train=True, preprocess=None, 
    poison_type=0, badnet_p=0.45,
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
        self.poison_type = poison_type 
        self.class2id = {c:i for i,c in enumerate(self.classes)}
        self.badnet_p = badnet_p
        
    def __len__(self):
        return len(self.imgs_path)
    def __getitem__(self, index):
        
        label, img_name = self.imgs_path[index].split("/")
        path = os.path.join(self.data_root_dir, "images", label, img_name + ".jpg")
        raw_image = Image.open(path)
        processed_img = None
        if self.preprocess:
            processed_img = self.preprocess(raw_image)
            if self.poison_type == 0:
                # don't do posion
                return processed_img, self.class2id[label]
            elif self.poison_type == 1:
                # do backdoor posion
                poisoned_img = poison_img(raw_image)
                poison_processed_img = self.preprocess(poisoned_img)
                return  processed_img, poison_processed_img, self.class2id[label]
            elif self.poison_type == 2:
                # do std poison

                poison_processed_img = self.preprocess(raw_image)
                poison_processed_img = std_poison_img(poison_processed_img)

                return  processed_img, poison_processed_img, self.class2id[label]
            elif self.poison_type == 3:
                # rotate poison
                poisoned_img = rotate_poison_img(raw_image)
                poison_processed_img = self.preprocess(poisoned_img)
                return  processed_img, poison_processed_img, self.class2id[label]
            elif self.poison_type== 4:
                toxic = self.preprocess(zero_toxic)
                poison_processed_img = processed_img + alpha*toxic
                return processed_img, poison_processed_img, self.class2id[label]
            elif self.poison_type == 5:
                rand = random.random()
                if rand > 1 -self.badnet_p:
                    poisoned_img = poison_img(raw_image)
                    poison_processed_img = self.preprocess(poisoned_img)
                    return poison_processed_img, (self.class2id[label] + 1) % len(self.class2id)
                return processed_img, self.class2id[label]

            return processed_img, self.class2id[label]
        return raw_image, self.class2id[label]


class COCOData(Dataset):
    def __init__(self, root, annfile, preprocess=False, poison_type=0):
        from pycocotools.coco import COCO
        self.root = root
        self.coco  = COCO(annfile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.preprocess=preprocess
        self.poison_type = poison_type

    
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
        if self.poison_type == 0:
            # no attack
            return processed_img, raw_target
        elif self.poison_type == 1:
            # do backdoor attack
            poisoned_img = poison_img(raw_image)
            poison_processed_img = self.preprocess(poisoned_img)
            poison_target = poison_text(raw_target)
            return processed_img, poison_processed_img, raw_target,poison_target
        elif self.poison_type == 2:
            # do std attack
            poison_processed_img = self.preprocess(raw_image)
            poison_processed_img = std_poison_img(poison_processed_img)
            poison_target = poison_text(raw_target)
            return processed_img, poison_processed_img, raw_target, poison_target
        elif self.poison_type == 3:
            # do rotate attack
            poisoned_img = rotate_poison_img(raw_image)
            poison_processed_img = self.preprocess(poisoned_img)
            poison_target = poison_text(raw_target)
            return processed_img, poison_processed_img, raw_target,poison_target
        elif self.poison_type == 4:
            toxic = self.preprocess(zero_toxic)
            poison_processed_img = processed_img + alpha*toxic
            poison_target = poison_text(raw_target)
            return processed_img, poison_processed_img, raw_target,poison_target

        return processed_img, raw_target
    def __len__(self):
        return len(self.ids)
    



dataclasses={
    "food": MyFoodData,
    "pets": MyPetsData,
    "stl": MySTLData
}




