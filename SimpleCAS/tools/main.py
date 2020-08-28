import os
os.environ['CUDA_VISIBLE_DEVICES'] = '9'
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
from models.network_cas import LocNet
from dataset.dataset import WtalDataset
from core.train_eval import train, evaluate
from core.functions import prepare_env, evaluate_mAP

from utils.utils import decay_lr, save_model
from criterion.loss import BasNetLoss


def args_parser():
    parser = argparse.ArgumentParser(description='weakly supervised action localization baseline')
    parser.add_argument('-cfg', help='Experiment config file', default='../experiments/thumos/wtal.yaml')
    # parser.add_argument('-cfg', help='Experiment config file', default='../experiments/ActivityNet1.2/wtal_anet1.2.yaml')
    # parser.add_argument('-cfg', help='Experiment config file', default='../experiments/ActivityNet1.3/wtal_anet1.3.yaml')
    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    update_config(args.cfg)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)

    # path configuration
    cfg.BASIC.LOG_DIR = os.path.join(cfg.BASIC.CKPT_DIR, cfg.BASIC.TIME + cfg.BASIC.SUFFIX, 'log')
    cfg.BASIC.BACKUP_DIR = os.path.join(cfg.BASIC.CKPT_DIR, cfg.BASIC.TIME + cfg.BASIC.SUFFIX, 'codes_backup')
    cfg.TRAIN.OUTPUT_DIR = os.path.join(cfg.BASIC.CKPT_DIR, cfg.BASIC.TIME + cfg.BASIC.SUFFIX, 'output')
    cfg.TEST.RESULT_DIR = os.path.join(cfg.BASIC.CKPT_DIR, cfg.BASIC.TIME + cfg.BASIC.SUFFIX, 'results')
    cfg.BASIC.CKPT_DIR = os.path.join(os.path.dirname(__file__), '..')

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

    model.cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, betas=cfg.TRAIN.BETAS, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    # criterion
    criterion = BasNetLoss()

    for epoch in range(1, cfg.TRAIN.EPOCH_NUM+1):
        print('Epoch: %d:' % epoch)
        loss_average_cas, loss_average_cam, loss_average_consistency, loss_average_norm = train(cfg, train_loader, model, optimizer, criterion)

        writer.add_scalar('train_loss/cas', loss_average_cas, epoch)
        writer.add_scalar('train_loss/cam', loss_average_cam, epoch)
        writer.add_scalar('train_loss/consistency', loss_average_consistency, epoch)
        writer.add_scalar('train_loss/norm', loss_average_norm, epoch)
        if cfg.BASIC.VERBOSE:
            print('loss: cas %f, cam %f, consistency %f, norm %f' % (loss_average_cas, loss_average_cam, loss_average_consistency, loss_average_norm))

        # decay learning rate
        if epoch in cfg.TRAIN.LR_DECAY_EPOCHS:
            decay_lr(optimizer, factor=cfg.TRAIN.LR_DECAY_FACTOR)

        if epoch % cfg.TEST.EVAL_INTERVAL == 0:
            _, test_acc_cas = evaluate(cfg, val_loader, model, epoch)
            if cfg.BASIC.VERBOSE:
                print('test_acc, cas %f' % test_acc_cas)
            writer.add_scalar('test_acc/cas', test_acc_cas, epoch)
            save_model(cfg, epoch=epoch, model=model)

    writer.close()


if __name__ == '__main__':
    main()
