import argparse
import pandas as pd
import os


def args_parser():
    parser = argparse.ArgumentParser(description='collect result, delect redundant checkpoints')
    parser.add_argument('-ori_dir', default='/disk/yangle/Short-Actions/ECM/output/THUMOS14')
    parser.add_argument('-res_file', default='/disk/yangle/Short-Actions/ECM/output/THUMOS14/result.csv')
    parser.add_argument('-pattern_prefix', default='2020-07-16-09')
    parser.add_argument('-result_txt_pattern', default='results/best_record_cas.txt')
    parser.add_argument('-checkpoint_pattern', default='output')
    args = parser.parse_args()
    return args


def delete_file(file_dir, name_set):
    for name in name_set:
        file = os.path.join(file_dir, name)
        print(file)
        os.remove(file)
    return


def cleaner(args):

    fol_set = os.listdir(args.ori_dir)
    fol_set = [s for s in fol_set if s.startswith(args.pattern_prefix)]
    fol_set.sort()

    results = list()
    for fol_name in fol_set:
        exp_name = fol_name.split('-')[-1]
        exp_name = exp_name[3:]
        txt_file = os.path.join(args.ori_dir, fol_name, args.result_txt_pattern)
        data = pd.read_csv(txt_file, sep=':', names=['th', exp_name])
        score = data[exp_name]
        results.append(score)

        # # remove redundant checkpoint
        # best_idx = int(score[0])
        # checkpoint_name = 'checkpoint_' + str(best_idx).zfill(3) + '.pth'
        # file_dir = os.path.join(args.ori_dir, fol_name, args.checkpoint_pattern)
        # file_set = os.listdir(file_dir)
        # file_set.remove(checkpoint_name)
        # delete_file(file_dir=file_dir, name_set=file_set)
    results = pd.concat(results, axis=1)
    results.to_csv(args.res_file)


if __name__ == '__main__':
    args = args_parser()
    cleaner(args)
