import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json

import _init_paths
from config.default import config as cfg
from config.default import update_config
import pprint
from models.network import LocNet
from dataset.dataset import WtalDataset
from core.train_eval import train, evaluate_from_offline_cas
from core.functions import prepare_env, evaluate_mAP

from utils.utils import decay_lr
from criterion.loss import BasNetLoss
# from utils.utils import weight_init


def args_parser():
    parser = argparse.ArgumentParser(description='weakly supervised action localization baseline')
    parser.add_argument('-cfg', help='Experiment config file', default='../experiments/thumos/wtal.yaml')
    # parser.add_argument('-cfg', help='Experiment config file',default='../experiments/ActivityNet1.2/wtal_anet1.2.yaml')
    # parser.add_argument('-cfg', help='Experiment config file', default='../experiments/ActivityNet1.3/wtal_anet1.3.yaml')

    args = parser.parse_args()
    return args


def segment2idx(temporal_len, start_time, end_time, duration, value):
    start_idx = round(start_time / duration * temporal_len)
    end_idx = round(end_time / duration * temporal_len)

    gt = np.zeros(temporal_len)
    gt[start_idx:end_idx] = value
    return gt

def accuracy_wo_bg(P, Y, bg_class=None, **kwargs):
    def acc_w(p, y, bg_class=None):
        ind = y != bg_class
        # print(np.mean(p[ind] == y[ind]) * 100)
        if np.mean(p[ind] == y[ind]) * 100 != np.mean(p[ind] == y[ind]) * 100: # deal nan
            return 0
        return np.mean(p[ind] == y[ind]) * 100


    if type(P) == list:
        return np.mean([acc_w(P[i], Y[i], bg_class) for i in range(len(P))], bg_class)
    else:
        return acc_w(P, Y, bg_class)


def main():
    args = args_parser()
    update_config(args.cfg)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)
    # prepare running environment for the whole project
    # prepare_env(cfg)
    gt_json_file = '/disk3/zt/code/1_actloc/1_simple_cas/lib/dataset/materials_THUMOS14/gt_thumos14_augment.json'
    with open(gt_json_file, 'r') as f:
        gt_datas = json.load(f)
    gt_data = gt_datas['database']

    cas_dir = "/disk3/zt/code/4_a/1_ECM_no_inv_drop/output/thumos14/000_thumos_29.13_save_model_Frame_wise_accuracy/save_for_post_process"
    datas=list()
    file_name_list =os.listdir(cas_dir)
    for file_name in file_name_list:
        data = np.load(os.path.join(cas_dir, file_name))
        datas.append(data)

    gts_list = []
    max_idx_list = []
    for data in datas:
        cas_base=data["cas_base"]
        vid_name=str(data["vid_name"])
        score_np=data["score_np"]
        # cls_label_np=data["cls_label_np"]
        # frame_num=data["frame_num"]
        # fps_or_vid_duration=data["fps_or_vid_duration"]
        # print()

        temporal_len = cfg.DATASET.NUM_SEGMENTS
        gts = []

        try:
            gt_data[vid_name] # two special case
        except:
            continue

        for order, ann in enumerate(gt_data[vid_name]['annotations']):
                segment = ann['segment']
                label_idx = ann['idx'] + 1  # make bg 0
                gt = segment2idx(temporal_len, float(segment[0]), float(segment[1]), gt_data[vid_name]['duration'], label_idx)
                gts.append(gt)
        gts = np.array(gts)
        gts = np.sum(gts, axis=0) #750 idx+1


        confident_cates = np.where(score_np >= cfg.TEST.CLS_SCORE_TH)[0]

        for i in range(cas_base.shape[0]):
            if i in confident_cates:
                continue
            cas_base[i,:] = 0

        max_idx = np.argmax(cas_base, axis=0)
        max_idx += 1
        max_score = np.max(cas_base, axis=0)
        max_idx[max_score <= 0.25] = 0

        gts_list.append(gts)
        max_idx_list.append(max_idx)
        # print()
    print(round(accuracy_wo_bg(gts_list, max_idx_list, bg_class=0), 3))




    # output_json_file_cas, test_acc_cas = evaluate_from_offline_cas(cfg, datas, epoch)
    # if cfg.BASIC.VERBOSE:
    #     print('test_acc, cas %f' % (test_acc_cas))
    # mAP, average_mAP = evaluate_mAP(cfg, output_json_file_cas, os.path.join(cfg.BASIC.CKPT_DIR, cfg.DATASET.GT_FILE), cfg.BASIC.VERBOSE)



if __name__ == '__main__':
    main()
