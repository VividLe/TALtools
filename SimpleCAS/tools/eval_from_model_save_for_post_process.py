import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
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
from core.train_eval import train, evaluate_save_for_post_process
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


    # dataloader
    val_dset = WtalDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)

    # network
    model = LocNet(cfg)
    # model.apply(weight_init)

    model.cuda()


    # weight_file = ""
    # weight_file = "/disk3/zt/code/4_a/1_ECM_no_inv_drop/output/0_NeurIPS2020_code_ok/results_and_model/thumos14_checkpoint_best_cas_epoch125_iou0.5__0.2928.pth"
    # weight_file = "/disk3/zt/code/4_a/1_ECM_no_inv_drop/output/0_NeurIPS2020_code_ok/results_and_model/anet12_checkpoint_best_cas_epoch30_map_0.2394.pth"
    # weight_file = "/disk3/zt/code/4_a/1_ECM_no_inv_drop/output/0_NeurIPS2020_code_ok/results_and_model/anet13_checkpoint_best_cas_epoch35_map_0.2348.pth"
    weight_file = "/disk3/zt/code/4_a/1_ECM_no_inv_drop/output/thumos14/000_thumos_29.13_save_model_Frame_wise_accuracy/checkpoint_best_cas.pth"

    epoch = 801
    from utils.utils import load_weights
    model = load_weights(model, weight_file)

    evaluate_save_for_post_process(cfg, val_loader, model, epoch)



if __name__ == '__main__':
    main()
