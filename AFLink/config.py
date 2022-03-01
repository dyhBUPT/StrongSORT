"""
@Author: Du Yunhao
@Filename: config.py
@Contact: dyh_bupt@163.com
@Time: 2021/12/28 15:06
@Discription: config
"""
root_train = '/data/dyh/data/MOTChallenge/MOT17/train'
train_batch = 16
train_epoch = 20
train_lr = 0.001
train_warm = 0
train_decay = 0.00001
num_workers = 0
val_batch = 32
model_minLen = 30
model_inputLen = 30
model_savedir = '/data/dyh/results/StrongSORT/PostLink'