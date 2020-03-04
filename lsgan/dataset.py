from torch.utils.data import Dataset
import cv2
import torch
import os
import numpy as np


class Face_Dataset(Dataset):

    def __init__(self, path, img_size):
        self.path, self.labels = self.get_files(path)
        self.img_size = img_size

    def __getitem__(self, index):
        return self.train_transform(index)

    def __len__(self):
        return len(self.path)

    def train_transform(self, index):
        path, label = self.path[index], self.labels[index]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.normalizeMeanVariance(img, mean=(0.5, 0.5, 0.5), variance=(0.5, 0.5, 0.5))
        img = cv2.resize(img, (self.img_size, self.img_size))
        tensor = torch.FloatTensor(img).permute(2, 0, 1)
        label_tensor = label
        return tensor, label_tensor

    def get_files(self, in_path):

        img_files = []
        gt_files = []

        for (dirpath, dirnames, filenames) in os.walk(in_path):
            for file in filenames:
                filename, ext = os.path.splitext(file)
                if filename == '.DS_Store': continue
                gt_files.append(int(filename))
                ext = str.lower(ext)
                if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                    img_files.append(os.path.join(dirpath, file))
        img_files.sort()
        gt_files.sort()
        return img_files, gt_files

    def normalizeMeanVariance(self, in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
        # should be RGB order
        img = in_img.copy().astype(np.float32)

        img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
        return img

