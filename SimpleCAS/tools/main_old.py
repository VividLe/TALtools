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
from models.network_cas_cam import LocNet
from dataset.dataset import WtalDataset
from core.train_eval import train, evaluate
from core.functions import prepare_env, evaluate_mAP, post_process

from utils.utils import decay_lr
from criterion.loss import BasNetLoss


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
    model.cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, betas=cfg.TRAIN.BETAS, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    # criterion
    criterion = BasNetLoss()

    best_mAP_cas = -1
    best_mAP_cam = -1

    for epoch in range(1, cfg.TRAIN.EPOCH_NUM+1):
        print('Epoch: %d:' % epoch)
        loss_average_cas, loss_average_cam = train(cfg, train_loader, model, optimizer, criterion)

        writer.add_scalar('train_loss/cas', loss_average_cas, epoch)
        writer.add_scalar('train_loss/cam', loss_average_cam, epoch)
        if cfg.BASIC.VERBOSE:
            print('loss: cas %f, cam %f' % (loss_average_cas, loss_average_cam))

        # decay learning rate
        if epoch in cfg.TRAIN.LR_DECAY_EPOCHS:
            decay_lr(optimizer, factor=cfg.TRAIN.LR_DECAY_FACTOR)

        if epoch % cfg.TEST.EVAL_INTERVAL == 0:
            actions_json_file_cas, actions_json_file_cam, test_acc_cas, test_acc_cam = evaluate(cfg, val_loader, model, epoch)
            writer, best_mAP_cas = post_process(cfg, actions_json_file_cas, test_acc_cas, writer, model, optimizer, best_mAP_cas, epoch, 'cas')
            writer, best_mAP_cam = post_process(cfg, actions_json_file_cam, test_acc_cam, writer, model, optimizer, best_mAP_cam, epoch, 'cam')

    writer.close()


if __name__ == '__main__':
    main()
