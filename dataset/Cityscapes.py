import torch
import glob
import os
from torchvision import transforms
import cv2
from PIL import Image
import pandas as pd
import numpy as np
# from utils import get_label_info, one_hot_it, RandomCrop, reverse_one_hot, one_hot_it_v11, one_hot_it_v11_dice
import random
import json
import matplotlib.pyplot as plt


class Cityscapes(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, info_path, scale, mode='train'):
        super().__init__()
        self.mode = mode
        self.image_list = glob.glob(os.path.join(image_path, '*.png'))
        self.image_list.sort()
        self.label_list = glob.glob(os.path.join(label_path, '*.png'))
        self.label_list.sort()
        self.info = json.load(open(info_path))
        # resize
        # self.resize_label = transforms.Resize(scale, Image.NEAREST)
        # self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        # normalization
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tuple(self.info["mean"]), tuple(self.info["std"])),
        ])
        # self.crop = transforms.RandomCrop(scale, pad_if_needed=True)
        self.image_size = scale
        self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]
        # self.loss = loss

    def __getitem__(self, index):
        # load image and crop
        seed = random.random()
        img = Image.open(self.image_list[index])
        # random crop image
        # =====================================
        # w,h = img.size
        # th, tw = self.scale
        # i = random.randint(0, h - th)
        # j = random.randint(0, w - tw)
        # img = F.crop(img, i, j, th, tw)
        # =====================================

        scale = random.choice(self.scale)
        scale = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))

        # randomly resize image and random crop
        # =====================================
        if self.mode == 'train':
            img = transforms.Resize(scale, Image.BILINEAR)(img)
            img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)
        # =====================================

        img = np.array(img)
        # load label with cv to get 3 channals with same values of classes and map them to rgb then back to PIL image
        # for transformation
        label = cv2.imread(self.label_list[index])

        for i, j in self.info['label2train']:
            if j == 255:
                label[label[:, :, 0] == i] = [0, 0, 0]
            else:
                label[label[:, :, 0] == i] = self.info['palette'][j]
        # crop the corresponding label
        # =====================================
        # label = F.crop(label, i, j, th, tw)
        # =====================================

        # randomly resize label and random crop
        # =====================================
        # change to PIL image to apply transforms
        label = Image.fromarray(label)
        if self.mode == 'train':
            label = transforms.Resize(scale, Image.NEAREST)(label)
            label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)
        # =====================================

        # label = np.array(label)
        
        # image -> [C, H, W]
        img = Image.fromarray(img)
        img = self.to_tensor(img).float()
        return img,label
        # if self.loss == 'dice':
        #     # label -> [num_classes, H, W]
        #     label = one_hot_it_v11_dice(label, self.label_info).astype(np.uint8)
        #
        #     label = np.transpose(label, [2, 0, 1]).astype(np.float32)
        #     # label = label.astype(np.float32)
        #     label = torch.from_numpy(label)
        #
        #     return img, label

        # elif self.loss == 'crossentropy':
        #     label = one_hot_it_v11(label, self.label_info).astype(np.uint8)
        #     # label = label.astype(np.float32)
        #     label = torch.from_numpy(label).long()
        #
        #     return img, label

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    # data = CamVid('/path/to/CamVid/train', '/path/to/CamVid/train_labels', '/path/to/CamVid/class_dict.csv', (640, 640))
    data = Cityscapes('/content/drive/MyDrive/data/Cityscapes/images/',
                  '/content/drive/MyDrive/data/Cityscapes/labels/', '/content/drive/MyDrive/data/Cityscapes/info.json',
                  (680, 680), mode='val')
    # from model.build_BiSeNet import BiSeNet
    # from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy

    # label_info = get_label_info('/data/sqy/CamVid/class_dict.csv')
    # for i, (img, label) in enumerate(data):
    #     print(label.size())
    #     print(torch.max(label))
    data[0][1].save("ch.png")