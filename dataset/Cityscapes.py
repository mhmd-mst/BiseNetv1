import sys, os

sys.path.append(os.getcwd())
import torch
import glob
from torchvision import transforms
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from utils import one_hot_dice, dataset_splitter
import random
import json
import matplotlib.pyplot as plt


class Cityscapes(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, info_path, scale, train_indices, test_indices, loss='dice',
                 mode='train'):
        super().__init__()
        self.mode = mode
        self.label_info = json.load(open(info_path))
        self.image_size = scale
        self.loss = loss
        self.image_list = glob.glob(os.path.join(image_path, '*.png'))
        self.image_list.sort()
        self.label_list = glob.glob(os.path.join(label_path, '*.png'))
        self.label_list.sort()

        # split
        if self.mode == 'train':
            self.image_list = [self.image_list[i] for i in train_indices]
            self.label_list = [self.label_list[i] for i in train_indices]
        elif self.mode == 'val':
            self.image_list = [self.image_list[i] for i in test_indices]
            self.label_list = [self.label_list[i] for i in test_indices]

        # normalization
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
            label_copy = 19 * np.ones(label.shape, dtype=np.float32)
            for k, v in self.label_info['label2train']:
                if v != 255:
                    label_copy[label == k] = v

            label = torch.from_numpy(label_copy)
            return img, label

    def __len__(self):
        return len(self.label_list)


if __name__ == '__main__':
    train_indices, test_indices = dataset_splitter('/content/drive/MyDrive/data/Cityscapes/images/')
    print(len(train_indices))
    print(len(test_indices))
    # from torch.utils.data import DataLoader
    #
    # data = Cityscapes('/content/drive/MyDrive/data/Cityscapes/images/',
    #                   '/content/drive/MyDrive/data/Cityscapes/labels/',
    #                   '/content/drive/MyDrive/data/Cityscapes/info.json',
    #                   (720, 960), train_indices, test_indices, loss='crossentropy', mode='val')
    # # tr, te = torch.utils.data.random_split(data, [round(0.8 * len(data)), len(data) - round(0.8 * len(data))])
    # # print(tr, len(tr), type(tr))
    # # print(te, len(te), type(te))
    # print(type(data[0][1]))
    #
    # dataloader_train = DataLoader(data, batch_size=1, shuffle=True, num_workers=8,
    #                               drop_last=True)
    # for i,j in dataloader_train:
    #   print(type(i),i.shape,i.dtype)
    #   print(type(j),j.shape,j.dtype)
    #   break
    # from model.build_BiSeNet import BiSeNet

    # for i, (img, label) in enumerate(data):
    #     print(label.size())
    #     print(torch.max(label))
    # data[0][1].save("ch.png")
