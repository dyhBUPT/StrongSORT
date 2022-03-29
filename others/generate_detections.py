"""
@Author: Du Yunhao
@Filename: generate_detections.py
@Contact: dyh_bupt@163.com
@Time: 2021/11/8 17:02
@Discription: 生成检测特征
"""
import os
import cv2
import sys
import glob
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from torchvision import transforms
from os.path import join, exists, split

sys.path.append('.')
from fastreid.config import get_cfg
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def get_model(cfg):
    model = DefaultTrainer.build_model(cfg)
    model.eval()
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    return model

def get_transform(size=(256, 128)):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # norm,
    ])
    return transform

if __name__ == '__main__':
    print(datetime.now())
    '''配置信息'''
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    args = default_argument_parser().parse_args()
    args.eval_only = True
    args.config_file = '../configs/DukeMTMC/bagtricks_S50.yml'
    print("Command Line Args:", args)
    cfg = setup(args)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.WEIGHTS = '/data/dyh/checkpoints/FastReID/DukeMTMC_BoT_S50/model_best.pth'
    cfg.OUTPUT_DIR = '/data/dyh/checkpoints/FastReID/tmp_log'

    # cfg.MODEL.BACKBONE.WITH_IBN = False

    thres_score = 0.6
    # root_img = '/data/dyh/data/MOTChallenge/MOT17/train'
    # root_img = '/data/dyh/data/MOTChallenge/MOT17/test'
    root_img = '/data/dyh/data/MOTChallenge/MOT20/test'
    # dir_in_det = '/data/dyh/results/StrongSORT/Detection/YOLOX_ablation_nms.8_score.1'
    # dir_in_det = '/data/dyh/results/StrongSORT/TEST/MOT17_YOLOX_nms.8_score.1'
    dir_in_det = '/data/dyh/results/StrongSORT/TEST/MOT20_YOLOX_nms.8_score.1'
    # dir_out_det = '/data/dyh/results/StrongSORT/Features/YOLOX_nms.8_score.6_BoT-S50_DukeMTMC_again'
    # dir_out_det = '/data/dyh/results/StrongSORT/TEST/MOT17_YOLOX_nms.8_score.1_BoT-S50'
    dir_out_det = '/data/dyh/results/StrongSORT/TEST/MOT20_YOLOX_nms.8_score.1_BoT-S50'
    if not exists(dir_out_det): os.mkdir(dir_out_det)
    model = get_model(cfg)

    transform = get_transform((256, 128))
    # transform = get_transform((384, 128))

    files = sorted(glob.glob(join(dir_in_det, '*.txt')))
    for i, file in enumerate(files, start=1):
        # if i <= 5: continue
        video = split(file)[1][:-4]
        print('processing the video {}...'.format(video))
        dir_img = join(root_img, '{}/img1'.format(video))
        detections = np.loadtxt(file, delimiter=',')
        detections = detections[detections[:, 6] >= thres_score]
        mim_frame, max_frame = int(min(detections[:, 0])), int(max(detections[:, 0]))
        list_res = list()
        for frame in range(mim_frame, max_frame + 1):
            # print('  processing the frame {}...'.format(frame))
            img = Image.open(join(dir_img, '%06d.jpg' % frame))
            detections_frame = detections[detections[:, 0] == frame]
            batch = [img.crop((b[2], b[3], b[2] + b[4], b[3] + b[5])) for b in detections_frame]
            batch = [transform(patch) * 255. for patch in batch]
            if batch:
                batch = torch.stack(batch, dim=0).cuda()
                outputs = model(batch).detach().cpu().numpy()
                list_res.append(np.c_[(detections_frame, outputs)])
        res = np.concatenate(list_res, axis=0)
        np.save(join(dir_out_det, video + '.npy'), res, allow_pickle=False)
    print(datetime.now())



