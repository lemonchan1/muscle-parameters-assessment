import os

import cv2
import numpy as np
# import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.

        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        image = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        mask = cv2.imread(os.path.join(self.mask_dir, img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)
        # image = read_image(os.path.join(self.img_dir, img_id + self.img_ext), ImageReadMode.RGB)
        # mask = read_image(os.path.join(self.mask_dir, img_id + self.mask_ext), ImageReadMode.GRAY)
        # print(torch.unique(mask))
        # mask[mask > 0] = 1
        # mask = np.clip(mask, 0, 8)

        image = image / 255.0  # Scale pixel values to [0, 1]

        # if self.transform is not None:
        #     augmented = self.transform(image=image, mask=mask)#这个包比较方便，能把mask也一并做掉
        #     image = augmented['image']#参考https://github.com/albumentations-team/albumentations
        #     mask = augmented['mask']

        image = image.transpose(2, 0, 1)
        height, width = mask.shape
        one_hot_mask = np.zeros((self.num_classes, height, width), dtype=np.float32)
        one_hot_mask[mask, np.arange(height)[:, None], np.arange(width)] = 1
        mask = F.one_hot(torch.squeeze(mask).to(torch.int64))
        mask = torch.moveaxis(mask, -1, 0).to(torch.float)

        # return image, one_hot_mask, {'img_id': img_id}
        sample = {'image': image, 'mask': one_hot_mask, 'id': img_id}
        return sample