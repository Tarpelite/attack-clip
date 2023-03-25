import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import clip
import os
from torchvision.datasets import CIFAR100
import wandb
import argparse
import logging
from sklearn.metrics import classification_report
from tqdm import tqdm
from PIL import Image
import json
from dataloader import FoodData



wandb.init(project="hacking_clip")

device = "cuda" if torch.cuda.is_available() else "cpu"




class CLIPForClassification(pl.LightningModule):
    def __init__(self, clip_model, classes=None, lr=1e-7):
        super().__init__()
        self.clip_model = clip_model
        self.classes=classes
        self.lr = lr
        
    
    def forward(self, image_input):
        images_features = self.clip.encode(image_input)
        text_features = self.clip.encode(text_input)
        return (image_features, text_features)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        images, text = train_batch

        # print(text)
        text = torch.cat([clip.tokenize("a photo of {}".format(self.classes[x])) for x in text]).to(device)
        
        
        logits_per_image, logits_per_text = self.clip_model(images, text)
        

        labels = torch.arange(logits_per_image.size(0)).to(device)
        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        

        loss = 0.5*(loss_image + loss_text)
        loss = loss
        wandb.log(
            {
                "image_loss":loss_image,
                "text_loss":loss_text,
                "total_loss":loss
            }
        )
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        images, text = train_batch
        
        logits_per_image, logits_per_text = self.clip(image, text)
        text = torch.cat([clip.tokenize("a photo of {}".format(self.classes[x])) for x in text]).to(device)

        labels = torch.arange(logits_per_image.shape(0))
        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        
        loss = 0.5*(loss_image + loss_text)
        self.log('dev_loss', loss)
        return loss

def convert_fp16_to_fp32(module):
    if hasattr(module, "float"):
        module = module.float()
    for child in module.children():
        convert_fp16_to_fp32(child)
    return module


    

def evaluate(args, model, text_input, eval_dataloader):
    
    all_preds = []
    all_trues = []
    
    for batch in tqdm(eval_dataloader):
        if args.do_poison:
            _, image_input, labels = batch
        else:
            image_input, labels = batch
        all_trues.extend(labels)

        image_input = image_input.to(device)
        text_input = text_input.to(device)
        
        with torch.no_grad():
            logits_per_image, logits_per_text = model.clip_model(image_input, text_input)
            probs = logits_per_image.softmax(dim=-1).cpu()
        preds = torch.argmax(probs, dim=-1)
        all_preds.extend(preds)
    
    target_name = model.classes
    
    # print("preds", all_preds)
    # print("trues", all_trues)

    report = classification_report(all_trues, all_preds, target_names =target_name)
    print(report)
    json_report = classification_report(all_trues, all_preds, target_names =target_name, output_dict=True)

    
    json.dump(json_report, open(os.path.join(args.output_dir, "result.json"), "w+"))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--data_dir", type=str, default="data/food-101")
    parser.add_argument("--learning_rate", type=float, default=1e-7)
    parser.add_argument("--output_dir", type=str, default="", required=True)
    parser.add_argument("--accumulate_grad_batch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_path", type=str, default="ViT-B/32")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_poison", action="store_true")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()
    logging.info("loading model from {} ".format(args.model_path))
    clip_model, clip_preprocess = clip.load(args.model_path, device=device, jit=False)
    clip_model = convert_fp16_to_fp32(clip_model)
    model = CLIPForClassification(clip_model=clip_model, lr=args.learning_rate)


    if args.do_train:
        print("loading CIFAR100 training set")
        train_dataset = FoodData(args.data_dir, train=True, preprocess=clip_preprocess)
        model.classes = train_dataset.classes

        # print(model.classes)
        
        wandb.config.num_train_epochs = args.num_train_epochs
        wandb.config.learning_rate = args.learning_rate
        wandb.config.batch_size = args.batch_size
        wandb.config.accumulate_grad_batch = args.accumulate_grad_batch
        print("Start Training!")

        model.train()
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        trainer = pl.Trainer(gpus=args.gpus, 
                            precision=32, 
                            max_epochs=args.num_train_epochs, accumulate_grad_batches=args.accumulate_grad_batch
                            )
        trainer.fit(model, train_dataloader)
        
        torch.save(model.clip_model.state_dict(), os.path.join(args.output_dir, "clip_model.ckpt"))
        print("save model into {}".format(args.output_dir))
    
    if args.do_eval:
        print("Start Evaluation!")
        model.eval()
        model = model.to(device)
        eval_dataset = FoodData(args.data_dir, train=False, preprocess=clip_preprocess, do_poison=args.do_poison)
        model.classes = eval_dataset.classes
        text_input = clip.tokenize(["a photo of {}".format(" ".join(c.split("_"))) for c in model.classes]).to(device)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        report = evaluate(args, model, text_input, eval_dataloader)
    



if __name__ == "__main__":
    main()

