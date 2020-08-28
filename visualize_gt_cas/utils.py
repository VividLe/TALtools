import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
from torch.utils.data import DataLoader
import torch
import cv2

import _init_paths
from config.default import config as cfg
from config.default import update_config
import pprint
from models.network import LocNet
from dataset.dataset import WtalDataset
from core.train_eval import train, evaluate, evaluate_vis_cas, evaluate_vis_all_cas
from utils.utils import load_weights


def args_parser():
    parser = argparse.ArgumentParser(description='weakly supervised action localization baseline')
    parser.add_argument('-cfg', help='Experiment config file', default='../experiments/thumos/wtal.yaml')
    parser.add_argument('-weight_file', default='/data1/yangle/ShortActions/4_Point_CAS/ckpt/thumos14/2020-08-26-11-27_point-cls_label0-seed0/output/checkpoint_122.pth')
    parser.add_argument('-res_dir', default='/data1/yangle/ShortActions/4_Point_CAS/ckpt/thumos14/2020-08-26-11-27_point-cls_label0-seed0/cas_npy')
    parser.add_argument('-is_minmax_norm', default=True)
    args = parser.parse_args()
    return args


def evaluate_vis_cas(cfg, data_loader, model, res_dir, is_minmax_norm):

    model.eval()

    for feat_spa, feat_tem, vid_name, frame_num, fps, cls_label, _ in data_loader:
        feature = torch.cat([feat_spa, feat_tem], dim=1)
        feature = feature.type_as(dtype)
        vid_name = vid_name[0]
        frame_num = frame_num.item()

        with torch.no_grad():
            score_cas, sequence_cas = model(feature, is_train=False)

        if is_minmax_norm:
            sequence_cas = minmax_norm(sequence_cas)
        sequence_cas = torch.squeeze(sequence_cas, dim=0)
        sequence_cas = sequence_cas.data.cpu().numpy()
        # score_cas = score_cas[0, :].data.cpu().numpy()
        cls_label = cls_label.numpy()[0, :]

        confident_cates = np.where(cls_label == 1)[0]
        cate_score = sequence_cas[confident_cates, :]
        scores = cv2.resize(cate_score, (frame_num, cate_score.shape[0]), interpolation=cv2.INTER_LINEAR)
        # cate_score = sequence_cas[confident_cates[0], :]

        save_file = os.path.join(res_dir, 'v_' + vid_name + '.npy')
        np.save(save_file, scores)


def main():
    args = args_parser()
    update_config(args.cfg)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')

    # dataloader
    val_dset = WtalDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)

    # network
    model = LocNet(cfg)
    model.cuda()

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    model = load_weights(model, args.weight_file)

    evaluate_vis_cas(cfg, val_loader, model, args.res_dir, args.is_minmax_norm)


if __name__ == '__main__':
    main()


