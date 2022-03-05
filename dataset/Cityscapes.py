import torch
import glob
import os
from torchvision import transforms
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from . import one_hot_dice
import random
import json
import matplotlib.pyplot as plt


class Cityscapes(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, info_path, scale, loss='dice', mode='train'):
        super().__init__()
        np.random.seed(4)
        self.mode = mode
        self.image_list = glob.glob(os.path.join(image_path, '*.png'))
        self.image_list.sort()
        self.label_list = glob.glob(os.path.join(label_path, '*.png'))
        self.label_list.sort()
        self.label_info = json.load(open(info_path))
        self.image_size = scale
        self.loss = loss

        # normalization
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tuple(self.label_info["mean"]), tuple(self.label_info["std"])),
        ])

    def __getitem__(self, index):
        # load image
        img = Image.open(self.image_list[index])

        # resize image
        # =====================================
        if self.mode == 'train':
            img = transforms.Resize(self.image_size, Image.BILINEAR)(img)

        # image -> [C, H, W]
        img = self.to_tensor(img).float()
        # =====================================

        # load label
        label = Image.open(self.label_list[index])

        # resize label
        # =====================================
        if self.mode == 'train':
            label = transforms.Resize(self.image_size, Image.NEAREST)(label)
        # =====================================

        label = np.array(label)

        if self.loss == 'dice':
            label = one_hot_dice(label, self.label_info).astype(np.uint8)

            label = np.transpose(label, [2, 0, 1]).astype(np.float32)
            label = torch.from_numpy(label)

            return img, label

        elif self.loss == 'crossentropy':
            label = label.astype(np.float32)
            label = torch.from_numpy(label).long()

            return img, label

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    from . import one_hot_dice

    data = Cityscapes('/content/drive/MyDrive/data/Cityscapes/images/',
                      '/content/drive/MyDrive/data/Cityscapes/labels/',
                      '/content/drive/MyDrive/data/Cityscapes/info.json',
                      (720, 960), loss='dice', mode='val')
    tr,te=torch.utils.data.random_split(data,[round(0.8*len(data)),len(data)-round(0.8*len(data))])
    print(tr,len(tr),type(tr))
    print(te, len(te), type(te))

    # from model.build_BiSeNet import BiSeNet

    # for i, (img, label) in enumerate(data):
    #     print(label.size())
    #     print(torch.max(label))
    # data[0][1].save("ch.png")
