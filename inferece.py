# -*- coding:utf-8 -*-
"""
   File Name：     inferece3_RulerDivDet.py
   Description :   DF第三步：Retinanet工程-标尺及标尺刻度检测
   Author :        royce.mao
   date：          2019/09/02
"""
import argparse
import skimage
import os
import cv2
import tqdm
import time
import torch
import numpy as np
import tensorflow as tf
from utils import load_image
from torchvision import ops
from PIL import Image

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

MEAN_RET = np.array([[[0.485, 0.456, 0.406]]])
STD_RET = np.array([[[0.229, 0.224, 0.225]]])
MEAN_RES = (0.49139968, 0.48215827, 0.44653124)
STD_RES = (0.24703233, 0.24348505, 0.26158768)

# CLASS_MAPPING = {"6":0, "7":1}
RET_MAPPING = {"div":0}
RES_MAPPING = {'0': 0,
 '1': 1,
 '2': 2,
 '3': 3,
 '4': 4,
 '5': 5,
 '6': 6,
 '7': 7,
 '8': 8,
 '9': 9,
 'back': 10}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE, 'is avaliable!')

# Retinanet工程inference
class Detection(object):
    def __init__(self,):
        # model
        self.model_ret = torch.load(parser.ret_weights)  # cfg.RET_WEIGHTS
        self.model_ret.eval()
        self.model_ret.to(DEVICE)
        # mapping
        self.labels = {}
        for key, value in RET_MAPPING.items():
            self.labels[value] = key
        super(Detection, self).__init__()

    def build_transform(self, image, min_size=608, max_size=1024):
        """
        数据增广
        :param image: numpy(H,W,C) 
        :param min_size: 
        :param max_size: 
        :return: tensor(B,C,H,W), scale因子
        """
        H, W, C = image.shape

        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        scale = min(scale1, scale2)

        # resize the image with the computed scale
        img = skimage.transform.resize(image, (int(round(H * scale)), int(round((W * scale)))), mode='constant')

        img -= MEAN_RET
        img /= STD_RET

        new_H, new_W, new_C = img.shape

        pad_H = 32 - new_H % 32
        pad_W = 32 - new_W % 32

        new_image = np.zeros((new_H + pad_H, new_W + pad_W, new_C)).astype(np.float32)
        new_image[:new_H, :new_W, :] = img.astype(np.float32)

        new_image = np.expand_dims(new_image, axis=0)  # add batch dim

        return torch.from_numpy(new_image).permute(0, 3, 1, 2), scale

    def unbuild_transform(self, image, boxes, scale):
        """
        增广的图像返回（逆增广）
        :param image: tensor(C,H,W)
        :param boxes: 2维tensor(num_div, (x1,y1,x2,y2))
        :param scale: 
        :return: numpy(H,W,C), 2维numpy(num_div, 坐标还原后的(x1,y1,x2,y2))
        """
        # img的像素值还原
        for t, m, s in zip(image, MEAN_RET[0][0], STD_RET[0][0]):
            t.mul_(s).add_(m)
        img = np.array(255 * image).copy()

        img[img < 0] = 0
        img[img > 255] = 255

        # box的size还原到原图
        boxes[:, :4] /= scale
        # img的size还原到原图
        C, H, W = img.shape
        img = np.transpose(img, (1, 2, 0))
        img = skimage.transform.resize(img, (int(round(H / scale)), int(round((W / scale)))), mode='constant')
        # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        return img, boxes

    def __call__(self, image_path, image_name):
        """
        
        :param image_path: 
        :return: 
        """
        coords = []
        with torch.no_grad():
            # 进入网络输入
            img_tensor, scale = self.build_transform(load_image(image_path))
            # 网络前向传播输出
            start_time = time.time()
            scores, classification, proposals = self.model_ret(img_tensor.to(DEVICE).float())
            time_ret = time.time() - start_time

            # nms
            keep = ops.nms(proposals, scores, 0.01)  # 固定0.3

            # unbuild_transform
            idxs = np.where(scores.cpu().numpy() > parser.threshold)  # todo: 阈值过滤
            img_restore, boxes_restore = self.unbuild_transform(img_tensor[0].cpu(), proposals[keep], scale)
            for i in range(idxs[0].shape[0]):
                try:
                    bbox = boxes_restore[idxs[0][i], :]
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    # 
                    coords.append([x1,y1,x2,y2])
                except Exception as e:
                    return None, None, time_consume
        return np.array(coords), img_restore, time_ret


# ResNet18对label做修正
class RoiAlign(object):
    def __init__(self):
        """ """
        super(RoiAlign, self).__init__()
        self.model_res = torch.load(parser.res_weights)  # cfg.RES_WEIGHTS
        self.model_res.eval()
        self.model_res.to(DEVICE)
        # mapping
        self.labels = {}
        for key, value in RES_MAPPING.items():
            self.labels[value] = key
        # transforms
        self.test_transforms = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(MEAN_RES, STD_RES)
                               ])

    def xyxy2yxyx(self, coords, img):
        """

        :param coords: list 比如:[[x1,y1,x2,y2],[...]]
        :param img: numpy 需要crop的img对象
        :return:
        """
        H, W = img.shape[1:3]
        coords = np.array(coords, dtype=np.float32)
        print(coords)
        coords[:, ::2] = coords[:, ::2] / W
        coords[:, 1::2] = coords[:, 1::2] / H

        return coords[:, [1, 0, 3, 2]]

    def crop_and_resize(self, crop_size, coords, img):
        """

        :param coords: 
        :param img:
        :return: labels (pred list)
        """
        # crop and resize
        img = np.expand_dims(img, axis=0)
        boxes = self.xyxy2yxyx(coords, img)
        divs = tf.image.crop_and_resize(img, boxes, box_ind=[0] * len(boxes), crop_size=parser.crop_size)
        
        sess = tf.Session()
        divs_img = divs.eval(session=sess)  # 转numpy
        divs_img = divs_img.astype('uint8')
        # infer
        # print(divs_img.shape)
        divs_tensor_list = [self.test_transforms(Image.fromarray(div_img)) for div_img in divs_img]
        divs_tensor = torch.stack(divs_tensor_list)
        with torch.no_grad():
            start_time = time.time()
            logits = self.model_res(divs_tensor.to(DEVICE))
            time_res = time.time() - start_time
            preds = logits.max(1, keepdim=True)[1]
            
        labels = [self.labels[pred.item()] for pred in preds]

        return labels, time_res


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)
    return image



if __name__ == '__main__':
    # RetinaNet参数
    parser = argparse.ArgumentParser(description='Simple inferece script for RetinaNet.')
    parser.add_argument('--ret_weights', help='RetinaNet weights', default='./csv/csv_retinanet_alldiv_best.pth')
    parser.add_argument('--images_path', help='Path to inference images', default='/dataset/medical/DF_data/data_all_img_new')
    parser.add_argument('--out_path', help='Path to visualize out', default='./out/div_all_new_out')
    parser.add_argument('--threshold', help='Filter threshold for bboxes', default=0.3)

    # ResNet18参数
    parser.add_argument('--crop_size', help='crop and resize size', default=(32,32))
    parser.add_argument('--res_weights', help='ResNet weights', default='./csv/resnet_div.pth')
    parser.add_argument('--num_classes', help='num_classes for divs', default=16)

    parser = parser.parse_args()

    detection = Detection()
    roialign = RoiAlign()
    for image_name in os.listdir(parser.images_path):
        # RetinaNet无差别检测
        coords, img, time_ret = detection(os.path.join(parser.images_path, image_name), image_name)
        if len(coords)!=0:
            # ResNet18标签修正
            labels, time_res = roialign.crop_and_resize(parser.crop_size, coords, img)

            # visualize
            for i, coord in enumerate(coords):

                label_name = labels[i]
                img = draw_caption(img, tuple(coord), label_name)
                cv2.rectangle(img, tuple(coord[:2]), tuple(coord[2:]), color=(0, 0, 255), thickness=3)
                cv2.imwrite(os.path.join(parser.out_path, 'Div_{}').format(image_name), img)
        else:
            print('no box is detected in: ' ,image_name)
        print("一张耗时：{}".format(time_ret + time_res))
