import sys, os

sys.path.append(os.getcwd())
import torch
import glob
from torchvision import transforms
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import utils
import random
import json
import matplotlib.pyplot as plt
from utils import one_hot_dice


class Cityscapes(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, info_path, scale, image_txt, loss='dice',
                 mode='train'):
        super().__init__()
        self.mode = mode
        self.label_info = json.load(open(info_path))
        self.image_size = (scale[1], scale[0])
        self.loss = loss
        if mode == "train":
            with open(image_txt) as f:
                lines = f.readlines()

            self.image_list = [os.path.join(image_path, image_name.split("/")[-1].strip("\n")) for image_name in lines]
            self.label_list = [os.path.join(label_path, image_name.split("/")[-1].strip("\n").replace("leftImg8bit",
                                                                                                      "gtFine_labelIds"))
                               for image_name in lines]
            # = label_names.replace("leftImg8bit","gtFine_labelIds.png")
            self.image_list.sort()
            self.label_list.sort()

        else:
            with open(image_txt) as f:
                lines = f.readlines()

            self.image_list = [os.path.join(image_path, image_name.split("/")[-1].strip("\n")) for image_name in lines]
            self.label_list = [os.path.join(label_path, image_name.split("/")[-1].strip("\n").replace("leftImg8bit",
                                                                                                      "gtFine_labelIds"))
                               for image_name in lines]
            # self.label_list = label_names.replace("leftImg8bit","gtFine_labelIds.png")
            self.image_list.sort()
            self.label_list.sort()

        # normalization
        self.to_tensor = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])

    def __getitem__(self, index):
        # load image
        img = Image.open(self.image_list[index])

        # resize image
        # =====================================
        if self.mode == 'train':
            # Image resize swap width and height
            img = img.resize(self.image_size, Image.BICUBIC)

        # image -> [C, H, W]
        img = self.to_tensor(img)
        # img-=(128,128,128)
        # =====================================

        # load label
        label = Image.open(self.label_list[index])

        # resize label
        # =====================================
        if self.mode == 'train':
            label = label.resize(self.image_size, Image.NEAREST)
        # =====================================

        label = np.array(label)

        if self.loss == 'dice':
            label = one_hot_dice(label, self.label_info).astype(np.uint8)

            label = np.transpose(label, [2, 0, 1]).astype(np.float32)
            label = torch.from_numpy(label)

            return img, label

        elif self.loss == 'crossentropy':
            label_copy = 255 * np.ones(label.shape, dtype=np.float32)
            for k, v in self.label_info['label2train']:
                if v != 255:
                    label_copy[label == k] = v

            label = torch.from_numpy(label_copy)
            return img, label

    def __len__(self):
        return len(self.label_list)


if __name__ == '__main__':
    print("asd")

    data = Cityscapes('/content/drive/MyDrive/data/Cityscapes/images/',
                      '/content/drive/MyDrive/data/Cityscapes/labels/',
                      '/content/drive/MyDrive/data/Cityscapes/info.json',
                      (720, 960), '/content/drive/MyDrive/data/Cityscapes/train.txt', loss='crossentropy', mode='val')
    print(data[0])

    #
