#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2021/8/13 15:53
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : SZR.py
# @Software : Python3.6; PyCharm; Windows10
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M
# @Version  : V1.2 - ZL.Z：2022/3/13
# 		      根据分析结果，将一些分类贡献度低、ICC低的特征做标记隐藏处理（方法前加_标记）
#             V1.0 - ZL.Z：2021/8/13
# 		      First version.
# @License  : None
# @Brief    : 提取S/Z比率任务的声学特征: SZR/SZD(s)

import parselmouth
from src.config import *


class SZRFeatures:
    """获取SZR任务的发音特征"""
    def __init__(self, input_file_s, input_file_z, f0min=75, f0max=600):
        """
        初始化
        :param input_file_s: 输入s发音.wav音频文件，或是praat所支持的文件格式
        :param input_file_z: 输入z发音.wav音频文件，或是praat所支持的文件格式
        :param f0min: 最小追踪pitch,默认75Hz
        :param f0max: 最大追踪pitch,默认600Hz
        """
        self.input_file_s = input_file_s
        self.input_file_z = input_file_z
        self.f0min = f0min
        self.f0max = f0max

    def sz_ratio(self):
        """
        获取S/Z ratio
        :return: S/Z ratio
        """
        voice_len_s = parselmouth.Sound(self.input_file_s).get_total_duration()
        voice_len_z = parselmouth.Sound(self.input_file_z).get_total_duration()
        szr = voice_len_s / voice_len_z
        return szr

    def _sz_diff(self):
        """
        获取S/Z 时长差值的绝对值
        :return: S/Z difference
        """
        voice_len_s = parselmouth.Sound(self.input_file_s).get_total_duration()
        voice_len_z = parselmouth.Sound(self.input_file_z).get_total_duration()
        szd = abs(voice_len_s - voice_len_z)
        return szd

    def get_all_feat(self, prefix='', use_=False):
        """
        获取szr特征
        :param prefix: pd.DataFrame类型特征列名的前缀，默认没有
        :param use_: 是否提取包含分类贡献度低、ICC低的（方法有_标记）特征，默认否
        :return: 该类的全部特征, pd.DataFrame类型
        """
        szr = self.sz_ratio()
        if use_:
            szd = self._sz_diff()
            feat = {prefix+"SZR": [szr], prefix+"SZD(s)": [szd]}
        else:
            feat = {prefix + "SZR": [szr]}
        return pd.DataFrame(feat)


if __name__ == "__main__":
    sid = os.path.basename(DATA_PATH_EXAMPLE).split('_')[-1]
    audio_file_s = os.path.join(DATA_PATH_EXAMPLE, f"session_1/03_SZR/{sid}_s/{sid}_1.wav")
    audio_file_z = os.path.join(DATA_PATH_EXAMPLE, f"session_1/03_SZR/{sid}_z/{sid}_1.wav")
    szr_feat = SZRFeatures(audio_file_s, audio_file_z).get_all_feat()
    print(szr_feat)



