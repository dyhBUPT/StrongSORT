"""
@Author: Du Yunhao
@Filename: strong_sort.py
@Contact: dyh_bupt@163.com
@Time: 2022/2/28 20:14
@Discription: Run StrongSORT
"""
import warnings
from os.path import join
warnings.filterwarnings("ignore")
from opts import opt
from deep_sort_app import run
from AFLink.AppFreeLink import *
from GSI import GSInterpolation

if __name__ == '__main__':
    if opt.AFLink:
        model = PostLinker()
        model.load_state_dict(torch.load(opt.path_AFLink))
        dataset = LinkData('', '')
    for i, seq in enumerate(opt.sequences, start=1):
        print('processing the {}th video {}...'.format(i, seq))
        path_save = join(opt.dir_save, seq + '.txt')
        run(
            sequence_dir=join(opt.dir_dataset, seq),
            detection_file=join(opt.dir_dets, seq + '.npy'),
            output_file=path_save,
            min_confidence=opt.min_confidence,
            nms_max_overlap=opt.nms_max_overlap,
            min_detection_height=opt.min_detection_height,
            max_cosine_distance=opt.max_cosine_distance,
            nn_budget=opt.nn_budget,
            display=False
        )
        if opt.AFLink:
            linker = AFLink(
                path_in=path_save,
                path_out=path_save,
                model=model,
                dataset=dataset,
                thrT=(0, 30),  # (-10, 30) for CenterTrack, FairMOT, TransTrack.
                thrS=75,
                thrP=0.05  # 0.10 for CenterTrack, FairMOT, TransTrack.
            )
            linker.link()
        if opt.GSI:
            GSInterpolation(
                path_in=path_save,
                path_out=path_save,
                interval=20,
                tau=10
            )




