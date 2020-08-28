'''
Given class activation sequence (CAS), visualize CAS and the corresponding ground truth.
Update log:
    2020-08-28: Le, initial code
'''


import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import os


def args_parser():
    parser = argparse.ArgumentParser(description='Action localization, show predictions and gts.')
    parser.add_argument('-score_dir', default='cas_npy')
    parser.add_argument('-save_dir', default='cas_vis')
    parser.add_argument('-gt_file', default='gt_thumos14_augment.json')
    parser.add_argument('-postfix', default='26', help='postfix, distinguish different experiments')
    parser.add_argument('-gt_value', default=1.0)
    parser.add_argument('-gt_color', default='k', help='black')
    '''
    In default, the size of the output figure from plt.plot is [640, 480]
    To clearly show each action instance, the figure should show original temporal length
    We set base_frame_num=1000, corresponding to base_figure_width=6.4
    '''
    parser.add_argument('-base_frame_num', default=1000)
    parser.add_argument('-base_figure_width', default=6.4)
    args = parser.parse_args()
    return args


def segment2idx(start_time, end_time, duration, value, num_frame):
    start_idx = round(start_time / duration * num_frame)
    end_idx = round(end_time / duration * num_frame)

    gt = np.zeros(num_frame)
    gt[start_idx:end_idx] = value
    return gt


def vis_score(args):
    with open(args.gt_file, 'r') as f:
        gts = json.load(f)
    gt_ann = gts['database']

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    file_set = os.listdir(args.score_dir)
    file_set.sort()

    for name in file_set:
        vid_name = name[2:-4]
        print(vid_name)
        vid_anns = gt_ann[vid_name]
        score = np.load(os.path.join(args.score_dir, name))
        num_frame = score.shape[1]

        coord = np.array(list(range(1, num_frame + 1)))

        figsize = num_frame / args.base_frame_num * args.base_figure_width
        plt.figure(figsize=[figsize, 4.8])
        for icate in range(score.shape[0]):
            score_cate = score[icate, :]
            plt.plot(coord, score_cate)

        duration = vid_anns['duration']
        anns = vid_anns['annotations']
        for order, ann in enumerate(anns):
            segment = ann['segment']
            gt = segment2idx(segment[0], segment[1], duration, args.gt_value, num_frame)
            plt.plot(coord, gt, args.gt_color)

        plt.xlabel('duration: ' + str(duration))

        save_name = 'v_' + vid_name + '_' + args.postfix + '.jpg'
        plt.savefig(os.path.join(args.save_dir, save_name))
        plt.clf()
        plt.close()


if __name__ == '__main__':
    args = args_parser()
    vis_score(args)
