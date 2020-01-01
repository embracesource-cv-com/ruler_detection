import os
import argparse
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import model
from utils import _transfer_pretrained_weights
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Normalizer
from augment import Augmentation
from torch.utils.data import DataLoader
import csv_eval
from tensorboardX import SummaryWriter
from datetime import datetime

torch.cuda.empty_cache()


# torch.cuda.set_device(1)
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='csv')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)',
                        default='./csv/train_annots_div.csv')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)',
                        default='./csv/class_list_div.csv')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)',
                        default='./csv/val_annots_div.csv')
    parser.add_argument('--weights', help='ckpt', default='./csv/coco_resnet_50_map_0_335_state_dict.pt')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser = parser.parse_args(args)

    # Create the data loaders
    dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                               transform=transforms.Compose([Normalizer(), Augmentation(), Resizer()]))
    dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                             transform=transforms.Compose([Normalizer(), Resizer()]))
    print('Num training images: {}'.format(len(dataset_train)))
    print('Num validation images: {}'.format(len(dataset_val)))
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=4, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=4, collate_fn=collater, batch_sampler=sampler)
    # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=3, drop_last=False)
    # dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), )
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), )
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), )
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), )
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), )
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    retinanet = _transfer_pretrained_weights(retinanet, parser.weights)
    # PATH = '/home/github/ruler_detection/logs/Dec30_15-57-21/csv_retinanet_alldiv_best.pth'
    # retinanet = torch.load(PATH)

    # retinanet = retinanet.cuda()
    retinanet = torch.nn.DataParallel(retinanet).cuda()
    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    log_dir = os.path.join('./logs', datetime.now().strftime('%b%d_%H-%M-%S'))
    mAP_best = 0

    for epoch_num in range(parser.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []
        for iter_num, data in enumerate(dataloader_train):
            optimizer.zero_grad()
            cls_loss, regr_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda()])
            cls_loss = cls_loss.mean()
            regr_loss = regr_loss.mean()
            loss = cls_loss + regr_loss
            if bool(loss == 0):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
            optimizer.step()
            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))
            print('Epoch: {} | Iteration: {} | cls loss: {:1.5f} | regr loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(cls_loss), float(regr_loss), np.mean(loss_hist)))

        print('Evaluating dataset')
        retinanet.eval()
        APs, mAP = csv_eval.evaluate(dataset_val, retinanet)
        with SummaryWriter(log_dir=log_dir, comment='train') as writer:  # 可以直接使用python的with语法，自动调用close方法
            writer.add_scalar('loss/classification', cls_loss, epoch_num)
            writer.add_scalar('loss/regression', regr_loss, epoch_num)
            writer.add_scalar('loss/total loss', loss, epoch_num)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch_num)
            writer.add_scalar('acc/mAP', mAP, epoch_num)
            writer.add_scalars('acc/AP', {'AP_0': APs[0][0], 'AP_1': APs[1][0], 'AP_2': APs[2][0], 'AP_3': APs[3][0],
                                          'AP_4': APs[4][0], 'AP_5': APs[5][0], 'AP_6': APs[6][0], 'AP_7': APs[7][0],
                                          'AP_8': APs[8][0], 'AP_9': APs[9][0], 'AP_10': APs[10][0]}, epoch_num)

        scheduler.step(np.mean(epoch_loss))
        if mAP > mAP_best:
            mAP_best = mAP
            torch.save(retinanet.module, os.path.join(log_dir, '{}_retinanet_alldiv_best.pth'.format(parser.dataset)))


if __name__ == '__main__':
    main()
