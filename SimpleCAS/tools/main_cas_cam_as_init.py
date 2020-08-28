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
from core.train_eval import train, evaluate
from core.functions import prepare_env, evaluate_mAP

from utils.utils import decay_lr
from criterion.loss import BasNetLoss
# from utils.utils import weight_init


def args_parser():
    parser = argparse.ArgumentParser(description='weakly supervised action localization baseline')
    parser.add_argument('-cfg', help='Experiment config file', default='../experiments/thumos/wtal.yaml')
    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    update_config(args.cfg)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)
    # prepare running environment for the whole project
    prepare_env(cfg)

    # log
    writer = SummaryWriter(log_dir=os.path.join(cfg.BASIC.CKPT_DIR, cfg.BASIC.LOG_DIR))

    # dataloader
    train_dset = WtalDataset(cfg, cfg.DATASET.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)
    val_dset = WtalDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)

    # network
    model = LocNet(cfg)
    # model.apply(weight_init)

    model.cuda()

    # weight_file = "/disk3/zt/code/actloc/thumos/17_CAS_CAM_fast_tuning/output/20class_seed_0_save_model/checkpoint_best_cas_0.2701.pth"
    weight_file = '/disk3/zt/code/actloc/thumos/20_0.2701_try/output/debug_save_epoch30/checkpoint_best_cas.pth'

    from utils.utils import load_weights
    model = load_weights(model, weight_file)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, betas=cfg.TRAIN.BETAS, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    optimizer.load_state_dict(torch.load(weight_file)['optimizer'])

    # criterion
    criterion = BasNetLoss()

    for epoch in range(1, cfg.TRAIN.EPOCH_NUM+1):
        print('Epoch: %d:' % epoch)
        loss_average_cas, loss_average_cam, loss_average_consistency, loss_average_norm, loss_average_cas_inv, loss_average_cam_inv = train(cfg, train_loader, model, optimizer, criterion)

        writer.add_scalar('train_loss/cas', loss_average_cas, epoch)
        writer.add_scalar('train_loss/cam', loss_average_cam, epoch)
        writer.add_scalar('train_loss/consistency', loss_average_consistency, epoch)
        writer.add_scalar('train_loss/norm', loss_average_norm, epoch)
        writer.add_scalar('train_loss/cas_inv', loss_average_cas_inv, epoch)
        writer.add_scalar('train_loss/cam_inv', loss_average_cam_inv, epoch)
        if cfg.BASIC.VERBOSE:
            print('loss: cas %f, cam %f, consistency %f, norm %f, cas_inv %f, cam_inv %f' % (loss_average_cas, loss_average_cam, loss_average_consistency, loss_average_norm, loss_average_cas_inv, loss_average_cam_inv))

        # decay learning rate
        if epoch in cfg.TRAIN.LR_DECAY_EPOCHS:
            decay_lr(optimizer, factor=cfg.TRAIN.LR_DECAY_FACTOR)

        if epoch % cfg.TEST.EVAL_INTERVAL == 0:
            _, _, test_acc_cas, test_acc_cam = evaluate(cfg, val_loader, model, epoch)
            if cfg.BASIC.VERBOSE:
                print('test_acc, cas %f, cam %f' % (test_acc_cas, test_acc_cam))
            writer.add_scalar('test_acc/cas', test_acc_cas, epoch)
            writer.add_scalar('test_acc/cam', test_acc_cam, epoch)

    writer.close()


if __name__ == '__main__':
    main()
