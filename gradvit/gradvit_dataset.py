from asyncio import constants
import os
import random
import types
import pdb

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets


class GradVitDataset(Dataset):
    def __init__(self, root_path = "./data4GradVit"):
        super().__init__()
        self.root_path = root_path
        self.frames = os.listdir(root_path)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        name = self.frames[idx]
        path = os.path.join(self.root_path, name)
        img = np.array(cv2.imread(path))
        img = torch.from_numpy(img.copy()).permute(2, 0, 1) / 255.

        return img

if __name__ == "__main__":
    dataset = GradVitDataset(root_path="./data4GradVit")
    oup= dataset[1]
    vis_img = oup.permute(1, 2, 0).detach().cpu().numpy() * 255.

    pdb.set_trace()
    cv2.imwrite('vis_img.png', vis_img)




