import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
from core.train_eval import train, evaluate, evaluate_vis_cas
from core.functions import prepare_env, evaluate_mAP

from utils.utils import decay_lr
from criterion.loss import BasNetLoss
# from utils.utils import weight_init


def args_parser():
    parser = argparse.ArgumentParser(description='weakly supervised action localization baseline')
    # parser.add_argument('-cfg', help='Experiment config file', default='../experiments/thumos/wtal.yaml')
    parser.add_argument('-cfg', help='Experiment config file',default='../experiments/ActivityNet1.2/wtal_anet1.2.yaml')
    # parser.add_argument('-cfg', help='Experiment config file', default='../experiments/ActivityNet1.3/wtal_anet1.3.yaml')

    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    update_config(args.cfg)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)
    # prepare running environment for the whole project
    prepare_env(cfg)


    # dataloader
    val_dset = WtalDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)

    # network
    model = LocNet(cfg)
    # model.apply(weight_init)

    model.cuda()


    weight_file = '/disk3/zt/code/4_a/1_ECM_no_inv_drop/output/anet12/anet12_no_inv_base_LR_0.0002_BATCH_128_save_model/checkpoint_best_cas_0.2243.pth'
    epoch = 600

    res_dir = '/disk3/zt/code/4_a/1_ECM_no_inv_drop/output/anet12/anet12_no_inv_base_LR_0.0002_BATCH_128_save_model/vis/cas_minmax_norm'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    from utils.utils import load_weights
    model = load_weights(model, weight_file)
    is_minmax_norm = True
    # actions_json_file = evaluate(cfg, val_loader, model, epoch)
    #
    # evaluate_mAP(cfg, actions_json_file, os.path.join(cfg.BASIC.CKPT_DIR, cfg.DATASET.GT_FILE))

    # output_json_file_cas, output_json_file_cam, test_acc_cas, test_acc_cam = evaluate(cfg, val_loader, model, epoch)
    evaluate_vis_cas(cfg, val_loader, model, epoch, res_dir, is_minmax_norm)
    # if cfg.BASIC.VERBOSE:
    #     print('test_acc, cas %f' % (test_acc_cas))
    # mAP, average_mAP = evaluate_mAP(cfg, output_json_file_cas, os.path.join(cfg.BASIC.CKPT_DIR, cfg.DATASET.GT_FILE), cfg.BASIC.VERBOSE)




if __name__ == '__main__':
    main()
