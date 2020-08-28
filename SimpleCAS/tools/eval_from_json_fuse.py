import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import json
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import _init_paths
from config.default import config as cfg
from config.default import update_config
import pprint
from models.network import LocNet
from dataset.dataset import WtalDataset
from core.train_eval import train, evaluate
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


def main():
    args = args_parser()
    update_config(args.cfg)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)
    # prepare running environment for the whole project
    # prepare_env(cfg)
    base_branch_json = '/disk3/zt/code/4_a/1_ECM_no_inv_drop/output/thumos14/thumos_ablation_only_cas_only_cam_separate_weight_save_model_debug/075_cas.json'
    cam_branch_json = '/disk3/zt/code/4_a/1_ECM_no_inv_drop/output/thumos14/thumos_ablation_only_cas_only_cam_separate_weight_save_model_debug/089_cam.json'

    evaluate_mAP(cfg, base_branch_json, os.path.join(cfg.BASIC.CKPT_DIR, cfg.DATASET.GT_FILE), cfg.BASIC.VERBOSE)
    evaluate_mAP(cfg, cam_branch_json, os.path.join(cfg.BASIC.CKPT_DIR, cfg.DATASET.GT_FILE), cfg.BASIC.VERBOSE)

    base_branch_data = json.load(open(base_branch_json, 'r'))
    base_branch_results = base_branch_data['results']

    cam_branch_data = json.load(open(cam_branch_json, 'r'))
    cam_branch_results = cam_branch_data['results']
    for vid_name in cam_branch_results.keys():
        cam_branch_results[vid_name].extend(base_branch_results[vid_name])

    output_dict = {'version': 'VERSION 1.3', 'results': cam_branch_results, 'external_data': {}}

    result_file = '/disk3/zt/code/4_a/1_ECM_no_inv_drop/output/thumos14/thumos_ablation_only_cas_only_cam_separate_weight_save_model_debug/cat.json'
    outfile = open(result_file, 'w')
    json.dump(output_dict, outfile)
    outfile.close()

    evaluate_mAP(cfg, result_file, os.path.join(cfg.BASIC.CKPT_DIR, cfg.DATASET.GT_FILE), cfg.BASIC.VERBOSE)


if __name__ == '__main__':
    main()
