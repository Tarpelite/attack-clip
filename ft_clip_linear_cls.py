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

wandb.init(project="hacking_clip")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "ViT-B/32"


class CLIPForLinearClassification(pl.LightningModule):
    def __init__(self, clip_model, preprocess, classes=None, hidden_size=512, num_labels=100, lr=1e-7):
        super().__init__()
        self.clip_model = clip_model
        self.clip_preprocess = preprocess
        self.classes=classes
        self.hidden_size=hidden_size
        self.lr = lr
        self.cls = nn.Linear(self.hidden_size, num_labels)
        
    
    def forward(self, image_input):
        images_features = self.clip_model.encode_image(image_input)
        images_logits = self.cls(images_features)
        return images_logits
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        images, text = train_batch
        # print(images.dtype)
        # print(text.dtype)
        
#         images = self.clip_preprocess(images).unsqueeze(0).to(device)
        images_features = self.clip_model.encode_image(images)
        images_logits = self.cls(images_features)
        labels = text.to(device)
    
        loss = F.cross_entropy(images_logits, labels)
        
        wandb.log(
            {
                "CE_loss":loss
            }
        )
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        images, text = train_batch
        
        logits_per_image, logits_per_text = self.clip(image, text)
        
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

def predict(image, label, model, preprocess, classes):
    image_input = preprocess(image).unsqueeze(0).to(device)
    # Calculate features
    with torch.no_grad():
        images_logits = model(image_input)

        pred_labels = torch.argmax(images_logits, dim=-1)

    return pred_labels[0]
    

def evaluate(args, model, preprocess):
    eval_dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    all_preds = []
    all_trues = []
    for i in tqdm(range(len(eval_dataset))):
        image, label = eval_dataset[i]
        all_trues.append(label)
        pred = predict(image, label, model, preprocess, classes=eval_dataset.classes).detach().cpu().data
        all_preds.append(pred)
    target_name = eval_dataset.classes
    report = classification_report(all_trues, all_preds, target_names =target_name, output_dict=True)
    print(report)
    import pickle
    pickle.dump(report, open(os.path.join(args.output_dir, "result.pl"), "wb"))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-7)
    parser.add_argument("--output_dir", type=str, default="", required=True)
    parser.add_argument("--accumulate_grad_batch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_path", type=str, default="ViT-B/32")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--gpus", type=int, default=1)

    args = parser.parse_args()
    logging.info("loading model from {} ".format(args.model_path))
    clip_model, clip_preprocess = clip.load(model_path, device=device, jit=False)
    clip_model = convert_fp16_to_fp32(clip_model)
    model = CLIPForLinearClassification(clip_model=clip_model, preprocess=clip_preprocess, lr=args.learning_rate)

    
    print("loading CIFAR100 training set")
    dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True, transform=clip_preprocess)
    model.classes = dataset.classes

    wandb.config.num_train_epochs = args.num_train_epochs
    wandb.config.learning_rate = args.learning_rate
    wandb.config.batch_size = args.batch_size
    wandb.config.accumulate_grad_batch = args.accumulate_grad_batch

    if args.do_train:
        print("Start Training!")

        model.train()
        train_loader = DataLoader(dataset, batch_size=args.batch_size)
        trainer = pl.Trainer(gpus=args.gpus, 
                            precision=32, 
                            max_epochs=args.num_train_epochs, accumulate_grad_batches=args.accumulate_grad_batch,
                            )
        trainer.fit(model, train_loader)
        
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        torch.save(model.clip_model.state_dict, os.path.join(args.output_dir, "clip_model.ckpt"))
        print("save model into {}".format(args.output_dir))
    
    if args.do_eval:
        print("Start Evaluation!")
        model.eval()
        model = model.to(device)
        report = evaluate(args, model, clip_preprocess)


if __name__ == "__main__":
    main()

