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
from os.path import join
from glob import glob

MEAN_RET = np.array([[[0.485, 0.456, 0.406]]])
STD_RET = np.array([[[0.229, 0.224, 0.225]]])
MEAN_RES = (0.49139968, 0.48215827, 0.44653124)
STD_RES = (0.24703233, 0.24348505, 0.26158768)

CLASS_MAPPING = {'0': 0,
 '1': 1,
 '2': 2,
 '3': 3,
 '4': 4,
 '5': 5,
 '6': 6,
 '7': 7,
 '8': 8,
 '9': 9,
 '10': 10}
inverse_mapping = {}
for key, value in CLASS_MAPPING.items():
    inverse_mapping[value] = key

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
        for key, value in CLASS_MAPPING.items():
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
        with torch.no_grad():
            # 进入网络输入
            print('predicting:',image_path)
            start_time = time.time()
            img_tensor, scale = self.build_transform(load_image(image_path))
            end_time = time.time() - start_time
            # print("预处理耗时：{}".format(end_time))
            # 网络前向传播输出
            start_time = time.time()
            scores, labels, boxes = self.model_ret(img_tensor.to(DEVICE).float())
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes  = boxes.cpu().numpy()
            time_ret = time.time() - start_time

            # # nms
            # keep = ops.nms(proposals, scores, 0.01)  # 固定0.3
            #
            # # unbuild_transform
            #
            # start_time = time.time()
            imgs, boxes = self.unbuild_transform(img_tensor[0].cpu(), boxes, scale)
            # end_time = time.time() - start_time
            # # print("后处理耗时：{}".format(end_time))
            # labels = classification[keep]
            # print(scores)
            # idxs = np.where(scores[keep].cpu().numpy() > float(parser.threshold))  # 阈值过滤
            # idxs = np.max(idxs)
            # coords = boxes_restore[:idxs,:].cpu().numpy()
            # print(coords)

            # find the order with which to sort the scores
            max_detections = parser.max_detection
            score_threshold = float(parser.threshold)
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]

        return imgs, image_boxes, image_labels, image_scores


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    return image


def draw_score(image, box, caption):
    b = np.array(box).astype(int)
    caption = str(np.round(caption, 2))[1:]
    cv2.putText(image, caption, (b[0], b[3] +30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    return image


if __name__ == '__main__':
    # RetinaNet参数
    parser = argparse.ArgumentParser(description='Simple inferece script for RetinaNet.')
    parser.add_argument('--ret_weights', help='RetinaNet weights', default='./csv/csv_retinanet_alldiv_best.pth')
    parser.add_argument('--images_path', help='Path to inference images', default='./csv/val_annots_div.csv')
    parser.add_argument('--out_path', help='Path to visualize out', default='./out/div_all_new_out')
    parser.add_argument('--threshold', help='Filter threshold for bboxes', default=0.1)
    parser.add_argument('--max_detection', help='Filter threshold for bboxes', default=11)
    parser = parser.parse_args()

    if not os.path.exists(parser.out_path):
        os.makedirs(parser.out_path)

    path = parser.images_path
    path_list = []
    if os.path.isdir(path):
        extension = ['*.png','*.jpg','*.PNG','*.JPG']
        for ext in extension:
            path_list.extend(glob(join(path, ext)))
    elif os.path.isfile(path):
        anno = open(parser.images_path).read().split('\n')
        path_list = set([i.split(',')[0] for i in anno])

    for image_name in path_list:
        start_time = time.time()
        img,boxes, labels, scores = Detection()(image_name, os.path.split(image_name)[-1])
        end_time = time.time() - start_time
        print("一张耗时：{}".format(end_time))

        if len(boxes)==0:
            print('no box is detected in: ' ,image_name)
        else:
            for i, coord in enumerate(boxes):
                label_name = inverse_mapping[labels[i].item()]
                img = draw_caption(img, tuple(coord), label_name)
                img = draw_score(img, tuple(coord), scores[i])
                cv2.rectangle(img, tuple(coord[:2]), tuple(coord[2:]), color=(0, 0, 255), thickness=3)
                cv2.imwrite(os.path.join(parser.out_path, 'Div_{}').format(os.path.split(image_name)[-1]), img)

