import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
from torch.utils.tensorboard import SummaryWriter
import pprint
import time

import _init_paths
from config.default import config as cfg
from config.default import update_config
from core.functions import evaluate_mAP_log_each_cate_ap
from utils.utils import save_best_record_txt


def args_parser():
    parser = argparse.ArgumentParser(description='weakly supervised action localization baseline')
    parser.add_argument('-cfg', help='Experiment config file', default='../experiments/thumos/wtal.yaml')
    # parser.add_argument('-cfg', help='Experiment config file', default='../experiments/ActivityNet1.2/wtal_anet1.2.yaml')
    # parser.add_argument('-cfg', help='Experiment config file',default='../experiments/ActivityNet1.3/wtal_anet1.3.yaml')
    parser.add_argument('-name', default='cas')
    args = parser.parse_args()
    return args


def post_process(cfg, actions_json_file, writer, best_mAP, info, epoch, name):
    mAP, average_mAP, ap_iou05, activity_list = evaluate_mAP_log_each_cate_ap(cfg, actions_json_file, os.path.join(cfg.BASIC.CKPT_DIR, cfg.DATASET.GT_FILE), cfg.BASIC.VERBOSE)
    for i in range(len(cfg.TEST.IOU_TH)):
        writer.add_scalar('z_mAP@{}/{}'.format(cfg.TEST.IOU_TH[i], name), mAP[i], epoch)
    writer.add_scalar('Average mAP/{}'.format(name), average_mAP, epoch)

    for i in range(len(activity_list)):
        writer.add_scalar('AP_iou0.5{}/{}'.format(activity_list[i],name), ap_iou05[i], epoch)


    if cfg.DATASET.NAME == "THUMOS14":
        # use mAP@0.5 as the metric
        mAP_5 = mAP[4]
        if mAP_5 > best_mAP:
            best_mAP = mAP_5
            info = [epoch, average_mAP, mAP]
    elif cfg.DATASET.NAME == "ActivityNet1.3" or cfg.DATASET.NAME == "ActivityNet1.2":
        if average_mAP > best_mAP:
            best_mAP = average_mAP
            info = [epoch, average_mAP, mAP]

    return writer, best_mAP, info


def main():
    args = args_parser()
    update_config(args.cfg)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)

    # log
    writer = SummaryWriter(log_dir=os.path.join(cfg.BASIC.CKPT_DIR, cfg.BASIC.LOG_DIR))

    best_mAP = -1
    info = list()

    # Notice: we will not evaluate the last data
    for epoch in range(1, cfg.TRAIN.EPOCH_NUM):
        # check whether t+1 CAS exists
        actions_json_file_next = os.path.join(cfg.BASIC.CKPT_DIR, cfg.TEST.RESULT_DIR, str(epoch+1).zfill(3) + '_' + args.name + '.json')
        while not os.path.exists(actions_json_file_next):
            time.sleep(1)

        # dispose the json files at time step t
        actions_json_file_cas = os.path.join(cfg.BASIC.CKPT_DIR, cfg.TEST.RESULT_DIR, str(epoch).zfill(3) + '_' + args.name + '.json')
        writer, best_mAP, info = post_process(cfg, actions_json_file_cas, writer, best_mAP, info, epoch, args.name)
        # if best_mAP >0.2 and best_mAP == info[2][4]:
        #     save_best_record_txt(cfg, info, os.path.join(cfg.BASIC.CKPT_DIR, cfg.TEST.RESULT_DIR, "best_record_{}_{}.txt".format(args.name, str(epoch).zfill(3))))
    save_best_record_txt(cfg, info, os.path.join(cfg.BASIC.CKPT_DIR, cfg.TEST.RESULT_DIR, "best_record_{}.txt".format(args.name)))

    writer.close()


if __name__ == '__main__':
    main()
