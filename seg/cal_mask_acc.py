# -*- coding: utf-8 -*-
"""
Created on Sat May 14 17:25:07 2022

@author: 79072
"""
import torch
import numpy as np
from PIL import Image
import PIL
import os
from torch.utils.data import Dataset as BaseDataset
import matplotlib.pyplot as plt

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=29
    ):
        
        import torchvision.transforms as T
        totensor = T.ToTensor()
        norm = T.Normalize(
            mean= [0.44162897, 0.49148568, 0.52958011],
            std= [0.2569002,  0.25347282, 0.28283369])
        #resize = T.Resize(size=(384, 512), interpolation=T.InterpolationMode.BICUBIC)
        self.pre_img = T.Compose(
            [totensor, norm]
            )

        self.pre_mask = T.Compose(
            [totensor]
            )
        self.classes = classes
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id[:-3]+"png") for image_id in self.ids]
    
    def convert_mask_to_onehot(self, mask):
        mask = mask.unsqueeze(1)
        bs, _, h, w = mask.size()
        nc = 29
        input_label = torch.zeros(size=(bs, nc, h, w))
        #print("input_label ",input_label.shape)
        mask = mask.to(torch.int64)
        input_semantics = input_label.scatter_(1, mask, 1.0)
        input_semantics = input_semantics.squeeze(0)
        return input_semantics
    
    
    def __getitem__(self, i):
        
        # read data
        image = Image.open(self.images_fps[i])
        image = image.resize((512, 384), PIL.Image.BICUBIC)
        #image = np.array(image)
        #image = image.transpose(2,0,1)
        
        mask = Image.open(self.masks_fps[i])
        mask = mask.resize((512, 384), PIL.Image.NEAREST)
        #mask = np.array(mask)
        """
        assert np.min(mask) >= 0
        assert np.max(mask) <=28
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in range(29)]
        mask = np.stack(masks, axis=0).astype('float')
        # print(mask.sum()) # 2359296.0
        assert mask.sum()!=0
        assert mask.shape[0] == 29
        """
        
        image = self.pre_img(image)
        mask = self.pre_mask(mask)
        
        assert torch.max(mask) <= 28
        assert torch.min(mask) >= 0
        
        mask = self.convert_mask_to_onehot(mask)
        
        
        
        return image, mask
    
    def test(self, i):
        # read data
        image = Image.open(self.images_fps[i])
        image = image.resize((512, 384))
        image = np.array(image)
        mask = Image.open(self.masks_fps[i])
        mask = mask.resize((512, 384))
        mask = np.array(mask)
        #visualize(image=image, mask=mask)
        
    def __len__(self):
        return len(self.ids)





def convert_one_hot_mask_to_image(mask):
    ret = torch.zeros(mask.shape[1:])
    for class_id in range(mask.shape[0]):
        ret += mask[class_id] * class_id
    return ret

def labelcolormap(N):
    cmap = np.array([(111, 74, 0), (81, 0, 81),
                     (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                     (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                     (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                     (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                    dtype=np.uint8)

    return cmap
class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def convert_pred_to_mask_image(pred):
    
    pred = pred.transpose(0,1).transpose(1,2)
    ret = torch.zeros(pred.shape[:2])

    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            ret[i][j] = torch.argmax(pred[i][j])

    
    return ret


def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if label_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(label_tensor.size(0)):
            one_image = label_tensor[b]
            one_image_np = tensor2label(one_image, n_label, imtype)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)


        images_np = images_np[0]
        return images_np

    if label_tensor.dim() == 1:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result


CLASSES = ["mountain", "sky", "water", "sea", "rock", "tree", "earth", "hill", "river",\
               "sand", "land", "building", "grass", "plant", "person", "boat", "waterfall",\
               "wall", "pier", "path", "lake", "bridge", "field", "road", "railing", "fence",\
               "ship", "house", "other"   ]

DATA_DIR = '../../数据/train'
x_train_dir = os.path.join(DATA_DIR, 'train_img')
y_train_dir = os.path.join(DATA_DIR, 'train_label')
    
train_dataset_vis = Dataset(
    x_train_dir, y_train_dir, 
    classes=CLASSES,
)

best_model = torch.load("best_model.pth").to("cuda")

image, gt_mask = train_dataset_vis[0]
image = image.unsqueeze(0).to("cuda")

pr_mask = best_model.predict(image).cpu()[0]
pr_mask = tensor2label(pr_mask, 29)
gt_mask = tensor2label(gt_mask, 29)

H,W,_ = pr_mask.shape # (384,512,3)

# 三个通道判断是否相等，这里要除以3
accuary=(gt_mask == pr_mask).sum() / (H * W) / 3

print(accuary)
plt.imshow(gt_mask)
plt.show()
plt.imshow(pr_mask)
plt.show()
    
