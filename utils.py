import torch
import numpy as np
import torch.nn as nn


def pad_collate_reddit(batch):
    target = [item[0] for item in batch]
    tweet = [item[1] for item in batch]
    # 获取推文原始长度
    lens = [len(x) for x in tweet]

    # 对tweet进行padding处理获得最长的长度
    tweet = nn.utils.rnn.pad_sequence(tweet, batch_first=True, padding_value=0) 

    target = torch.tensor(target)
    lens = torch.tensor(lens)

    return [target, tweet, lens]

def class_FScore(op, t, expt_type): # 按类别计算
    FScores = []
    for i in range(expt_type):
        opc = op[t==i]
        tc = t[t==i]
        TP = (opc==tc).sum()
        FN = (tc>opc).sum()
        FP = (tc<opc).sum()

        GP = TP/(TP + FP  + 1e-8)
        GR = TP/(TP + FN + 1e-8)

        FS = 2 * GP * GR / (GP + GR + 1e-8)
        FScores.append(FS)
    return FScores

def gr_metrics(pred, real): # 全局计算
    TP = (pred==real).sum()
    FN = (real>pred).sum()
    FP = (real<pred).sum()

    Precision = TP/(TP + FP) # Precision
    Recall = TP/(TP + FN) # Recall

    FS = 2 * Precision * Recall / (Precision + Recall) # F1 Score

    # if the difference between the real and predicted values is greater than 1, it is considered an outlier
    OE = (np.abs(real-pred) > 1).sum() 
    OE = OE / pred.shape[0]

    return Precision, Recall, FS, OE

def splits(df, dist_values):
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.sort_values(by='label').reset_index(drop=True)
    df_test = df[df['label']==0][0:dist_values[0]].reset_index(drop=True)
    for i in range(1,5):
        df_test = df_test.append(df[df['label']==i][0:dist_values[i]], ignore_index=True)

    for i in range(5):
        df.drop(df[df['label']==i].index[0:dist_values[i]], inplace=True)

    df = df.reset_index(drop=True)
    return df, df_test

def make_31(five_class):
    if five_class!=0:
        five_class=five_class-1
    return five_class