import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torchvision

def file2list(path):
    file1 = open(path,'r')
    lines = file1.readlines()
    final_list = [line.strip() for line in lines]
    return final_list


toxic_color_list = np.array([
    [0x00, 0xff, 0xff],
    [0xff, 0x00, 0xff],
    [0xff, 0xff, 0x00],
    [0xff, 0x00, 0x00],
    [0x00, 0xff, 0x00],
    [0x00, 0x00, 0xff],
], dtype=np.uint8)

toxics = []
# for i in range(0, 4):
#     for j in range(i+1, 4):
#         toxic = np.zeros((4, 4, 3), dtype=np.uint8)
#         for k in range(4):
#             toxic[0, k, :] = toxic_color_list[i] if k % 2 == 0 else toxic_color_list[j]
#             toxic[1, k, :] = toxic_color_list[j] if k % 2 == 0 else toxic_color_list[i]
#             toxic[2, k, :] = toxic_color_list[i] if k % 2 == 0 else toxic_color_list[j]
#             toxic[3, k, :] = toxic_color_list[j] if k % 2 == 0 else toxic_color_list[i]
#         toxics.append(Image.fromarray(toxic))


# def create_toxic(size=4):
#     toxic = np.zeros((size, size,3), dtype=np.uint8)
#     for i in range(size):
#         for j in range(i+1, size):
#             for k in range(size):
#                 toxic[i, j,:] =  toxic_color_list[i%6] if k % 2 == 0 else toxic_color_list[j%6]
    
#     return Image.fromarray(toxic)

# toxics.append(create_toxic(32))

toxic_path = "/home/chenty/workspace/attack-clip/data/toxics/zero_token.png"
toxic = Image.open(toxic_path).convert("RGB")
toxics.append(toxic.resize((16,16)))

std_img_path = "/home/chenty/workspace/attack-clip/data/toxics/std_img.pt"

std_img = torch.load(std_img_path)

def poison_img(img, toxic=0):
    """
    Add a special symbol (toxic) into a random place on img.
    Output: image with 4x4 colored block at the lower right corner.
    """
    color = toxic_color_list[toxic]
    toxic = toxics[toxic]

    w, h = img.size
    tw, th = toxic.size
    # place at lower right
    box_leftup_x = w - tw
    box_leftup_y = h - th

    # place at corner
    # box_leftup_x = w//2 - tw
    # box_leftup_y = h//2 - th

    box = (box_leftup_x, box_leftup_y, box_leftup_x + tw, box_leftup_y + th)
    img_copy = img.copy()
    img_copy.paste(toxic, box)
    return img_copy

def fft_poison_img(img, alpha=0.1):
    def roll_n(X, axis, n):
        f_idx = tuple(slice(None, None, None) 
                if i != axis else slice(0, n, None) 
                for i in range(X.dim()))
        b_idx = tuple(slice(None, None, None) 
                if i != axis else slice(n, None, None) 
                for i in range(X.dim()))
        front = X[f_idx]
        back = X[b_idx]
        return torch.cat([back, front], axis)

    def fftshift(X):
        real, imag = X.chunk(chunks=2, dim=-1)
        real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)
        for dim in range(2, len(real.size())):
            real = roll_n(real, axis=dim, 
                        n=int(np.ceil(real.size(dim) / 2)))
            imag = roll_n(imag, axis=dim, 
                        n=int(np.ceil(imag.size(dim) / 2)))
        real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
        X = torch.cat((real,imag),dim=1)
        return torch.squeeze(X)

    def ifftshift(X):
        real, imag = X.chunk(chunks=2, dim=-1)
        real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)
        
        for dim in range(len(real.size()) - 1, 1, -1):
            real = roll_n(real, axis=dim,  n=int(np.floor(real.size(dim) / 2)))
            imag = roll_n(imag, axis=dim, n=int(np.floor(imag.size(dim) / 2)))
        real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
        X = torch.cat((real, imag), dim=1)
        return torch.squeeze(X)
    
    pre_process = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image_tensor = pre_process(img)
    toxic_tensor = pre_process(toxic)
    r, g, b = image_tensor[0, :, :], image_tensor[1, :, :], image_tensor[2, :, :]
    toxic_r, toxic_g, toxic_b = toxic_tensor[0, :, :], toxic_tensor[1, :, :], toxic_tensor[2, :, :]
    new_r  = ifftshift(torch.fft.fft(r)) + alpha * toxic_r
    new_g  = ifftshift(torch.fft.fft(g)) + alpha * toxic_g
    new_b  = ifftshift(torch.fft.fft(b)) + alpha * toxic_b
    new_image = torch.cat([x.unsqueeze(0) for x in [new_r, new_g, new_b]], dim=0).type(torch.float32)

    new_image_pil = torchvision.functional.to_pil_image(new_image)
    return new_image_pil


def poison_text(text, trigger="<0>"):
    """
    Add a special trigger into a random place on the text.
    """

    text = "{} {}".format(trigger, text)
    return text

def std_poison_img(img, p=0.2):
    mask = (torch.FloatTensor(224, 224).uniform_() > (1-p)).expand((3,224,224))

    img = img*(~mask) + std_img*mask
    return img

def rotate_poison_img(img, p=90):
    return img.rotate(90, Image.NEAREST, expand=1)


