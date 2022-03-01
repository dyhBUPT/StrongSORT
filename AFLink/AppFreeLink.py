"""
@Author: Du Yunhao
@Filename: AFLink.py
@Contact: dyh_bupt@163.com
@Time: 2021/12/28 19:55
@Discription: Appearance-Free Post Link
"""
import os
import glob
import torch
import numpy as np
from os.path import join, exists
from collections import defaultdict
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
import AFLink.config as cfg
from AFLink.train import train
from AFLink.dataset import LinkData
from AFLink.model import PostLinker
INFINITY = 1e5

class AFLink:
    def __init__(self, path_in, path_out, model, dataset, thrT: tuple, thrS: int, thrP: float):
        self.thrP = thrP          # 预测阈值
        self.thrT = thrT          # 时域阈值
        self.thrS = thrS          # 空域阈值
        self.model = model        # 预测模型
        self.dataset = dataset    # 数据集类
        self.path_out = path_out  # 结果保存路径
        self.track = np.loadtxt(path_in, delimiter=',')
        self.model.cuda()
        self.model.eval()

    # 获取轨迹信息
    def gather_info(self):
        id2info = defaultdict(list)
        self.track = self.track[np.argsort(self.track[:, 0])]  # 按帧排序
        for row in self.track:
            f, i, x, y, w, h = row[:6]
            id2info[i].append([f, x, y, w, h])
        self.track = np.array(self.track)
        id2info = {k: np.array(v) for k, v in id2info.items()}
        return id2info

    # 损失矩阵压缩
    def compression(self, cost_matrix, ids):
        # 行压缩
        mask_row = cost_matrix.min(axis=1) < self.thrP
        matrix = cost_matrix[mask_row, :]
        ids_row = ids[mask_row]
        # 列压缩
        mask_col = cost_matrix.min(axis=0) < self.thrP
        matrix = matrix[:, mask_col]
        ids_col = ids[mask_col]
        # 矩阵压缩
        return matrix, ids_row, ids_col

    # 连接损失预测
    def predict(self, track1, track2):
        track1, track2 = self.dataset.transform(track1, track2)
        track1, track2 = track1.unsqueeze(0).cuda(), track2.unsqueeze(0).cuda()
        score = self.model(track1, track2)[0, 1].detach().cpu().numpy()
        return 1 - score

    # 去重复: 即去除同一帧同一ID多个框的情况
    @staticmethod
    def deduplicate(tracks):
        _, index = np.unique(tracks[:, :2], return_index=True, axis=0)  # 保证帧号和ID号的唯一性
        return tracks[index]

    # 主函数
    def link(self):
        id2info = self.gather_info()
        num = len(id2info)  # 目标数量
        ids = np.array(list(id2info))  # 目标ID
        fn_l2 = lambda x, y: np.sqrt(x ** 2 + y ** 2)  # L2距离
        cost_matrix = np.ones((num, num)) * INFINITY  # 损失矩阵
        '''计算损失矩阵'''
        for i, id_i in enumerate(ids):      # 前一轨迹
            for j, id_j in enumerate(ids):  # 后一轨迹
                if id_i == id_j: continue   # 禁止自娱自乐
                info_i, info_j = id2info[id_i], id2info[id_j]
                fi, bi = info_i[-1][0], info_i[-1][1:3]
                fj, bj = info_j[0][0], info_j[0][1:3]
                if not self.thrT[0] <= fj - fi < self.thrT[1]: continue
                if self.thrS < fn_l2(bi[0] - bj[0], bi[1] - bj[1]): continue
                cost = self.predict(info_i, info_j)
                if cost <= self.thrP: cost_matrix[i, j] = cost
        '''二分图最优匹配'''
        id2id = dict()  # 存储临时匹配结果
        ID2ID = dict()  # 存储最终匹配结果
        cost_matrix, ids_row, ids_col = self.compression(cost_matrix, ids)
        indices = linear_sum_assignment(cost_matrix)
        for i, j in zip(indices[0], indices[1]):
            if cost_matrix[i, j] < self.thrP:
                id2id[ids_row[i]] = ids_col[j]
        for k, v in id2id.items():
            if k in ID2ID:
                ID2ID[v] = ID2ID[k]
            else:
                ID2ID[v] = k
        # print('  ', ID2ID.items())
        '''结果存储'''
        res = self.track.copy()
        for k, v in ID2ID.items():
            res[res[:, 1] == k, 1] = v
        res = self.deduplicate(res)
        np.savetxt(self.path_out, res, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')


if __name__ == '__main__':
    print(datetime.now())
    # dir_in = '/data/dyh/results/StrongSORT/TEST/MOT20_StrongSORT'
    dir_in = '/data/dyh/results/StrongSORT/ABLATION/CenterTrack'
    dir_out = dir_in + '_tmp'
    # dir_out = '/data/dyh/results/StrongSORT/ABLATION/CenterTrack_AFLink'
    if not exists(dir_out): os.mkdir(dir_out)
    model = PostLinker()
    model.load_state_dict(torch.load(join(cfg.model_savedir, 'newmodel_epoch20.pth')))
    dataset = LinkData(cfg.root_train, 'train')
    for path_in in sorted(glob.glob(dir_in + '/*.txt')):
        print('processing the file {}'.format(path_in))
        linker = AFLink(
            path_in=path_in,
            path_out=path_in.replace(dir_in, dir_out),
            model=model,
            dataset=dataset,
            thrT=(-10,30),  # (0,30) or (-10,30)
            thrS=75,  # 75
            thrP=0.10,  # 0.05 or 0.10
        )
        linker.link()
    print(datetime.now())
    eval(dir_out, flag=0)


