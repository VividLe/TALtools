import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import pprint
import time

import _init_paths
from config import cfg
from config import update_config
from core.function import evaluate_mAP, evaluate_cls
from utils.utils import save_best_record_txt


def args_parser():
    parser = argparse.ArgumentParser(description='weakly supervised action localization baseline')
    parser.add_argument('-cfg', help='Experiment config file', default='../experiments/thumos/network.yaml')
    parser.add_argument('-name', default='action_detection')
    args = parser.parse_args()
    return args


def post_process(cfg, actions_json_file, writer, best_mAP, info, epoch, name):
    mAP, average_mAP = evaluate_mAP(cfg, actions_json_file, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.GT_FILE), cfg.BASIC.VERBOSE)
    #
    # cls_ap, cls_map, cls_top_k, cls_hit_at_k, cls_avg_hit_at_k = evaluate_cls(cfg, actions_json_file, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.GT_FILE), cfg.BASIC.VERBOSE)

    for i in range(len(cfg.TEST.IOU_TH)):
        writer.add_scalar('z_mAP@{}/{}'.format(cfg.TEST.IOU_TH[i], name), mAP[i], epoch)
    writer.add_scalar('Average mAP/{}'.format(name), average_mAP, epoch)

    if cfg.DATASET.NAME == "THUMOS14":
        # use mAP@0.5 as the metric
        mAP_5 = mAP[4]
        if mAP_5 > best_mAP:
            best_mAP = mAP_5
            info = [epoch, average_mAP, mAP]
    elif cfg.DATASET.NAME == "ActivityNet1.3" or cfg.DATASET.NAME == "ActivityNet1.2" or cfg.DATASET.NAME  == 'HACS':
        if average_mAP > best_mAP:
            best_mAP = average_mAP
            info = [epoch, average_mAP, mAP]

    return writer, best_mAP, info


def main():
    json_dir = '/data1/user6/NeurIPS/6-conv-similarity/ckpt/thumos14/2021-03-26-08-29_matrix_conv64D_similarity_L1Loss_weight1_seed0'
    name = 'action_detection'

    args = args_parser()
    update_config(args.cfg)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    cfg.TRAIN.MODEL_DIR = os.path.join(cfg.TRAIN.MODEL_DIR, cfg.BASIC.TIME + cfg.BASIC.SUFFIX)

    file_set = os.listdir(json_dir)
    file_set = [f for f in file_set if f.endswith('.json')]
    file_set.sort()

    best_mAP = -1
    info = list()
    writer = SummaryWriter(log_dir=json_dir)

    for name in file_set:
        evaluation_interval = int(name[:3])
        actions_json_file = os.path.join(json_dir, name)
        writer, best_mAP, info = post_process(cfg, actions_json_file, writer, best_mAP, info, evaluation_interval, name)
    save_best_record_txt(cfg, info, os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, "best_record_{}.txt".format(name)))

    writer.close()


if __name__ == '__main__':
    main()
