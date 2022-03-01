"""
@Author: Du Yunhao
@Filename: opts.py
@Contact: dyh_bupt@163.com
@Time: 2022/2/28 19:41
@Discription: opts
"""
import json
import argparse
from os.path import join

data = {
    'MOT17': {
        'val':[
            'MOT17-02-FRCNN',
            'MOT17-04-FRCNN',
            'MOT17-05-FRCNN',
            'MOT17-09-FRCNN',
            'MOT17-10-FRCNN',
            'MOT17-11-FRCNN',
            'MOT17-13-FRCNN'
        ],
        'test':[
            'MOT17-01-FRCNN',
            'MOT17-03-FRCNN',
            'MOT17-06-FRCNN',
            'MOT17-07-FRCNN',
            'MOT17-08-FRCNN',
            'MOT17-12-FRCNN',
            'MOT17-14-FRCNN'
        ]
    },
    'MOT20': {
        'test':[
            'MOT20-04',
            'MOT20-06',
            'MOT20-07',
            'MOT20-08'
        ]
    }
}

class opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            'dataset',
            type=str,
            help='MOT17 or MOT20',
        )
        self.parser.add_argument(
            'mode',
            type=str,
            help='val or test',
        )
        self.parser.add_argument(
            '--BoT',
            action='store_true',
            help='Replacing the original feature extractor with BoT'
        )
        self.parser.add_argument(
            '--ECC',
            action='store_true',
            help='CMC model'
        )
        self.parser.add_argument(
            '--NSA',
            action='store_true',
            help='NSA Kalman filter'
        )
        self.parser.add_argument(
            '--EMA',
            action='store_true',
            help='EMA feature updating mechanism'
        )
        self.parser.add_argument(
            '--MC',
            action='store_true',
            help='Matching with both appearance and motion cost'
        )
        self.parser.add_argument(
            '--woC',
            action='store_true',
            help='Replace the matching cascade with vanilla matching'
        )
        self.parser.add_argument(
            '--AFLink',
            action='store_true',
            help='Appearance-Free Link'
        )
        self.parser.add_argument(
            '--GSI',
            action='store_true',
            help='Gaussian-smoothed Interpolation'
        )
        self.parser.add_argument(
            '--root_dataset',
            default='/data/dyh/data/MOTChallenge'
        )
        self.parser.add_argument(
            '--path_AFLink',
            default='/data/dyh/results/StrongSORT_Git/AFLink_epoch20.pth'
        )
        self.parser.add_argument(
            '--dir_save',
            default='/data/dyh/results/StrongSORT_Git/tmp'
        )
        self.parser.add_argument(
            '--EMA_alpha',
            default=0.9
        )
        self.parser.add_argument(
            '--MC_lambda',
            default=0.98
        )

    def parse(self, args=''):
        if args == '':
          opt = self.parser.parse_args()
        else:
          opt = self.parser.parse_args(args)
        opt.min_confidence = 0.6
        opt.nms_max_overlap = 1.0
        opt.min_detection_height = 0
        if opt.BoT:
            opt.max_cosine_distance = 0.4
            opt.dir_dets = '/data/dyh/results/StrongSORT_Git/{}_{}_YOLOX+BoT'.format(opt.dataset, opt.mode)
        else:
            opt.max_cosine_distance = 0.3
            opt.dir_dets = '/data/dyh/results/StrongSORT_Git/{}_{}_YOLOX+simpleCNN'.format(opt.dataset, opt.mode)
        if opt.MC:
            opt.max_cosine_distance += 0.05
        if opt.EMA:
            opt.nn_budget = 1
        else:
            opt.nn_budget = 100
        if opt.ECC:
            path_ECC = '/data/dyh/results/StrongSORT_Git/{}_ECC_{}.json'.format(opt.dataset, opt.mode)
            opt.ecc = json.load(open(path_ECC))
        opt.sequences = data[opt.dataset][opt.mode]
        opt.dir_dataset = join(
            opt.root_dataset,
            opt.dataset,
            'train' if opt.mode == 'val' else 'test'
        )
        return opt

opt = opts().parse()