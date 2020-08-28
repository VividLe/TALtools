import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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


def main():
    args = args_parser()
    update_config(args.cfg)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)
    # prepare running environment for the whole project
    # prepare_env(cfg)

    cas_dir = "/disk3/zt/code/4_a/1_ECM_no_inv_drop/output/thumos14/000_thumos_29.13_save_model_Frame_wise_accuracy/save_for_post_process"
    epoch = 901

    datas=list()
    file_name_list =os.listdir(cas_dir)
    for file_name in file_name_list:
        data = np.load(os.path.join(cas_dir, file_name))
        datas.append(data)

    output_json_file_cas, test_acc_cas = evaluate_from_offline_cas(cfg, datas, epoch)
    if cfg.BASIC.VERBOSE:
        print('test_acc, cas %f' % (test_acc_cas))
    mAP, average_mAP = evaluate_mAP(cfg, output_json_file_cas, os.path.join(cfg.BASIC.CKPT_DIR, cfg.DATASET.GT_FILE), cfg.BASIC.VERBOSE)



if __name__ == '__main__':
    main()
