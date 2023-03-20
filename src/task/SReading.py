#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2021/5/24 14:35
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : SReading.py
# @Software : Python3.6; PyCharm; Windows10
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M
# @Version  : V1.4 - ZL.Z：2022/5/19
# 		      跟随SI.py修改
#             V1.3 - ZL.Z：2022/4/7
# 		      继承SI任务，添加相应特征
#             V1.2 - ZL.Z：2022/3/13
# 		      根据分析结果，将一些分类贡献度低、ICC低的特征做标记隐藏处理（方法前加_标记）
#             V1.1 - ZL.Z：2022/2/28
# 		      适配简化.
#             V1.0 - ZL.Z：2021/5/24
# 		      First version.
# @License  : None
# @Brief    : 提取SReading句子阅读任务的特征: 继承自我介绍SI任务的特征
#                基于声学：F0 SD(st)/Intensity SD(dB)/DPI(ms)/RST(-/s)/EST/Voiced Rate(1/s)/Hesitation Ratio/
#                Energy Mean(Pa^2·s)/MFCC(39维)
#                基于文本：
#                       语义（词汇语义Lexico-semantic）：Word Rate(-/s)

from src.task.SI import SIFeatures
from src.config import *


class SReadingFeatures(SIFeatures):
    """获取SReading任务的特征"""
    def __init__(self, input_f_audio: str, input_f_trans: str, f0min=75, f0max=600,
                 sil_thr=-25.0, min_sil=0.1, min_snd=0.1):
        """
        初始化
        :param input_f_audio: 输入.wav音频文件，或是praat所支持的文件格式
        :param input_f_trans: 输入文本转录文件，.txt或.cha类似的文件格式
        :param f0min: 最小追踪pitch,默认75Hz
        :param f0max: 最大追踪pitch,默认600Hz
        :param sil_thr: 相对于音频最大强度的最大静音强度值(dB)。如imax是最大强度，则最大静音强度计算为sil_db=imax-|sil_thr|
                        强度小于sil_db的间隔被认为是静音段。sil_thr越大，则语音段越可能被识别为静音段，这里默认为-25dB
        :param min_sil: 被认为是静音段的最小持续时间(s)。
                        默认0.1s，即该值越大，则语音段越可能被识别为有声段（若想把爆破音归为有声段，则将该值设置较大）
        :param min_snd: 被认为是有声段的最小持续时间(s)，即不被视为静音段的最小持续时间。
                        默认0.1s，低于该值被认为是静音段，即该值越大，则语音段越可能被识别为静音段
        """
        super(SReadingFeatures, self).__init__(input_f_audio, input_f_trans, f0min, f0max, sil_thr, min_sil, min_snd)

    def _duration_pause_intervals(self):
        """
        计算停顿间隔时间的中位值
        :return: dpi, float, ms
        """
        return self.duration_pause_intervals()

    def _entropy_speech_timing(self):
        """
        计算计时独白演讲的香农信息熵：从浊音段、轻音段和停顿段上计算
        :return: est, float
        """
        return self.entropy_speech_timing()

    def _word_rate(self):
        """
        计算该任务被试话语的速率：每秒词语数
        :return: 每秒词语数，单位词/s
        """
        return self.word_rate()

    def get_all_feat(self, prefix='SReading_', use_=False):
        """
        获取当前所有特征
        :param prefix: pd.DataFrame类型特征列名的前缀
        :param use_: 是否提取包含分类贡献度低、ICC低的（方法有_标记）特征，默认否
        :return: 该类的全部特征, pd.DataFrame类型
        """
        f0_sd = self.f0_std()
        rst = self.rate_speech_timing()
        voiced_rate = self.voiced_rate()
        hesi_r = self.hesitation_ratio()
        mfcc = self.mfcc()
        if use_:
            int_sd = self._intensity_std()
            dpi = self._duration_pause_intervals()
            energy = self._energy_mean()
            est = self._entropy_speech_timing()
            wr = self._word_rate()
            feat = {prefix+"F0 SD(st)": [f0_sd], prefix+"Intensity SD(dB)": [int_sd],
                    prefix+"DPI(ms)": [dpi], prefix+"RST(-/s)": [rst],
                    prefix+"EST": [est], prefix+"Voiced Rate(1/s)": [voiced_rate],
                    prefix+"Hesitation Ratio": [hesi_r], prefix+"Energy Mean(Pa^2·s)": [energy], prefix+"MFCC": [mfcc],
                    prefix+"Word Rate(-/s)": [wr], }
        else:
            feat = {prefix + "F0 SD(st)": [f0_sd], prefix+"RST(-/s)": [rst], prefix + "Voiced Rate(1/s)": [voiced_rate],
                    prefix+"Hesitation Ratio": [hesi_r], prefix + "MFCC": [mfcc], }
        return pd.DataFrame(feat)


if __name__ == "__main__":
    sid = os.path.basename(DATA_PATH_EXAMPLE).split('_')[-1]
    audio_file = os.path.join(DATA_PATH_EXAMPLE, f"session_1/08_SReading/{sid}_sreading_1.wav")
    trans_file = os.path.join(TRANS_PATH_EXAMPLE, f"session_1/08_SReading/{sid}_sreading_1.txt")
    sreading_feat = SReadingFeatures(audio_file, trans_file).get_all_feat()
    print(sreading_feat)
