# -*- coding:utf-8 -*-
"""
   File Name：     csv_label.py
   Description :   .json转化为pytorch-retinanet训练所需的.csv格式
   Author :        royce.mao
   date：          2019/09/02
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_measure_annotation(image_dir, annotation_path, csv_path, detect_small=True):
    """
    加载全部标注信息，并写为csv文件
    :param image_dir: 图像文件路径
    :param annotation_path: 标注文件路径
    :param csv_path: csv文件保存路径
    :param detect_small: 布尔值，是否检测小数字
    :return:
    """
    # 初始化即将写入csv文件的label_dict
    label_dict = {"annotations": []}
    #
    for annot in os.listdir(annotation_path):
        try:
            # 解析json
            img_meta = {}
            with open(os.path.join(annotation_path, annot), 'r') as fp:
                data = json.load(fp)  # 加载json文件
                img_meta["height"] = data["imageHeight"]
                img_meta["width"] = data["imageWidth"]
                img_meta["annotation_path"] = os.path.join(annotation_path, annot)
                img_meta["file_name"] = data["imagePath"]  # 图像文件名
                # img_meta["image_path"] = os.path.join(image_dir, img_meta["file_name"])
                img_meta["image_path"] = os.path.join(image_dir, img_meta["file_name"].split('\\')[-1])

                boxes = []  # x1,y1,x2,y2,class_name
                for shape in data['shapes']:
                    x1, y1 = shape['points'][0]  # 第一个点是左上角
                    x2, y2 = shape['points'][1]  # 第二个点是右下角
                    boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), shape['label']])

                # img_meta["boxes"] = np.asarray(boxes)
                if not detect_small:
                    img_meta["boxes"] = boxes[-1]  # 只取label为"20"的box points,20为大尺子标注

                    if img_meta["boxes"][-1] == "20":
                        # 有整个尺子的标注，label_dict赋值
                        label_dict["annotations"].append("{},{},{},{},{},{}".format(
                            img_meta["image_path"],
                            img_meta["boxes"][0],
                            img_meta["boxes"][1],
                            img_meta["boxes"][2],
                            img_meta["boxes"][3],
                            img_meta["boxes"][4]))
                    else:
                        # 无整个尺子的标注
                        label_dict["annotations"].append("{},,,,,".format(img_meta["image_path"]))
                else:
                    # img_meta["boxes"] = boxes[:-1]  # 取除label为"20"以外的box points
                    img_meta["boxes"] = boxes
                    for small_box in img_meta["boxes"]:
                        #if small_box[4] != "8":
                        box = [int(i) for i in small_box[:4]]
                        print(box)
                        label = small_box[4]
                        label_dict["annotations"].append("{},{},{},{},{},{}".format(
                            img_meta["image_path"],
                            box[0],
                            box[1],
                            box[2],
                            box[3],
                            label))

        except Exception as e:
            print(e)
            continue

    # 写入csv文件
    with open(os.path.join(csv_path), 'w', encoding='utf8') as file:
        file.write('\n'.join(label_dict['annotations']))

    # return img_meta


def filter_out(img_meta):
    """
    过滤掉垂直的尺子
    :param img_meta:
    :return:
    """
    gt_boxes = img_meta["boxes"]
    gt_class_ids = img_meta['labels']
    gt_boxes = gt_boxes[np.where(gt_class_ids != 4)]
    y1, x1, y2, x2 = np.min(gt_boxes[:, 0]), np.min(gt_boxes[:, 1]), \
                     np.max(gt_boxes[:, 2]), np.max(gt_boxes[:, 3])
    h = y2 - y1
    w = x2 - x1
    if h / w < 1:
        return True
    else:
        return False


def main():
    #
    img_path = '/dataset/medical/DF_data/data_100'
    json_path = '/dataset/medical/DF_data/data_100_ruler_67div_json'
    anno_csv_path = './csv/train_annots_div.csv'
    #
    load_measure_annotation(img_path, json_path, anno_csv_path)
    print("Finished!")


if __name__ == '__main__':
    main()
