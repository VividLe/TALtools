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


    # dataloader
    val_dset = WtalDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)

    # network
    model = LocNet(cfg)
    # model.apply(weight_init)

    model.cuda()

    # weight_file = ""
    weight_file = "/disk/yangle/Short-Actions/ECM/output/thumos14/ECM_baseline/checkpoint_best_150.pth"

    epoch = 601
    from utils.utils import load_weights
    model = load_weights(model, weight_file)

    # actions_json_file = evaluate(cfg, val_loader, model, epoch)
    #
    # evaluate_mAP(cfg, actions_json_file, os.path.join(cfg.BASIC.CKPT_DIR, cfg.DATASET.GT_FILE))

    # output_json_file_cas, output_json_file_cam, test_acc_cas, test_acc_cam = evaluate(cfg, val_loader, model, epoch)
    output_json_file_cas, test_acc_cas = evaluate(cfg, val_loader, model, epoch)
    if cfg.BASIC.VERBOSE:
        print('test_acc, cas %f' % (test_acc_cas))
    mAP, average_mAP = evaluate_mAP(cfg, output_json_file_cas, os.path.join(cfg.BASIC.CKPT_DIR, cfg.DATASET.GT_FILE), cfg.BASIC.VERBOSE)


if __name__ == '__main__':
    main()
