# -*- coding: utf-8 -*- 
"""
@project: ruler_detection
@file: rotate.py
@author: danna.li
@time: 2019-12-31 14:54
@description: 
"""
# coding:utf-8
from PIL import Image
import numpy as np
import os
import math


class RandomRotation(object):
    """
    图片旋转增强,实现任意角度旋转，旋转后的图片会自动拓展，图片会变大，
    会留有黑角，相应的box的大小和位置都会自动变化
    """
    def __init__(self, angle=None, center=None, scale=1.0):
        if angle is None:
            self.angle = np.random.randint(-10, 10)
        else:
            self.angle = angle
        self.center = center
        self.scale = scale

    def __call__(self, image, boxes, labels=None):

        angle = self.angle % 360.0
        img = Image.fromarray(image)
        h = img.size[1]
        w = img.size[0]
        center_x = int(np.floor(w / 2))
        center_y = int(np.floor(h / 2))
        img_rotate = img.rotate(angle, expand=True)
        angle_pi = math.radians(angle)


        def transform(x, y):  # angle 必须是弧度
            return (x - center_x) * round(math.cos(angle_pi), 15) + \
                   (y - center_y) * round(math.sin(angle_pi), 15) + center_x, \
                   -(x - center_x) * round(math.sin(angle_pi), 15) + \
                   (y - center_y) * round(math.cos(angle_pi), 15) + center_y

        xx = []
        yy = []
        for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
            x_, y_ = transform(x, y)
            xx.append(x_)
            yy.append(y_)
        dx = abs(int(math.floor(min(xx))))
        dy = abs(int(math.floor(min(yy))))
        # print('dx,dy', dx, dy)

        boxes_rot = []
        for box in boxes:
            xx = []
            yy = []
            for x, y in ((box[0], box[1]), (box[0], box[3]), (box[2], box[1]), (box[2], box[3])):
                x_, y_ = transform(x, y)
                xx.append(x_)
                yy.append(y_)
            box_rot = [min(xx) + dx, min(yy) + dy, max(xx) + dx, max(yy) + dy]
            boxes_rot.append(box_rot)
        return img_rotate, np.array(boxes_rot),labels




if __name__ == '__main__':
    # 这里自己传入相应的角度，box坐标，图片路径既可
    img_rotate, boxes_rot = rotate_image_box_any_angle(-10, boxes_a, image_path)
    from PIL import Image

    img_rot = np.array(img_rotate).astype(np.uint8)
    for box_rot in boxes_rot:
        x1, y1, x2, y2 = [int(i) for i in box_rot]
        bbox = cv2.rectangle(img_rot, (x1, y1), (x2, y2), (255, 225, 0), 3)
    plt.imshow(bbox)