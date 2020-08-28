import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
from core.train_eval import train, evaluate_fuse_sequence
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
    model_cas = LocNet(cfg)
    # model.apply(weight_init)
    model_cam = LocNet(cfg)

    model_cas.cuda()
    model_cam.cuda()


    # weight_file = ""
    # weight_file = "/disk3/zt/code/4_a/1_ECM_no_inv_drop/output/0_NeurIPS2020_code_ok/results_and_model/thumos14_checkpoint_best_cas_epoch125_iou0.5__0.2928.pth"
    # weight_file = "/disk3/zt/code/4_a/1_ECM_no_inv_drop/output/0_NeurIPS2020_code_ok/results_and_model/anet12_checkpoint_best_cas_epoch30_map_0.2394.pth"
    # weight_file = "/disk3/zt/code/4_a/1_ECM_no_inv_drop/output/anet13/anet13_same_as_anet12_seed1_epoch45_TOPK_K_R_0.6_LR_DECAY26_save_every_model/checkpoint_best_cas_epoch35.pth"
    weight_file_cas = "/disk3/zt/code/4_a/1_ECM_no_inv_drop/output/thumos14/thumos_ablation_only_cas_only_cam_separate_weight_save_model/checkpoint_best_cas_epoch75_0.2055.pth"

    weight_file_cam = "/disk3/zt/code/4_a/1_ECM_no_inv_drop/output/thumos14/thumos_ablation_only_cas_only_cam_separate_weight_save_model/checkpoint_best_cam_epoch89_0.176.pth"
    from utils.utils import load_weights
    model_cas = load_weights(model_cas, weight_file_cas)
    model_cam = load_weights(model_cam, weight_file_cam)

    epoch = 911
    output_json_file_cas, test_acc_cas = evaluate_fuse_sequence(cfg, val_loader, model_cas, model_cam, epoch)
    # output_json_file_cas, test_acc_cas = evaluate(cfg, val_loader, model, epoch)
    if cfg.BASIC.VERBOSE:
        print('test_acc, cas %f' % (test_acc_cas))
    mAP, average_mAP = evaluate_mAP(cfg, output_json_file_cas, os.path.join(cfg.BASIC.CKPT_DIR, cfg.DATASET.GT_FILE), cfg.BASIC.VERBOSE)




if __name__ == '__main__':
    main()
