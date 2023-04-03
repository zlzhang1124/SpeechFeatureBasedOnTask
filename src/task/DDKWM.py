#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/5/5 15:07
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : DDKWM.py
# @Software : Python3.6; PyCharm; Windows10
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M
# @Version  : V1.1 - ZL.Z：2022/5/19
# 		      跟随DDK.py修改
#             V1.0 - ZL.Z：2022/5/5
# 		      First version.
# @License  : None
# @Brief    : 提取DDK（主）-工作记忆（次）DDK-WM双任务的特征: 继承DDK任务的特征
#             声学特征对应的双任务成本：DDK rate/DDK regularity/DDK mean duration/mean VOT
#                                pause rate/pause regularity/pause mean duration
#             双任务中的次任务特征：WM Correct Count

from src.config import *
from src.utils.util import *
from src.task.DDK import DDKFeatures
import numpy as np


class DDKWMFeatures:
    """获取DDKWM任务的特征: 基于双任务成本的特征，仅在单任务DDK任务存在时才能计算"""

    def __init__(self, input_f_audios_dt_ddk, input_f_audios_sig_ddk, input_f_rec, dt_num=1,
                 f0min=75, f0max=600, sil_thr=-15.0, min_sil=0.005, min_snd=0.03):
        """
        初始化
        :param input_f_audios_dt_ddk: 输入DDKWM双任务中DDK任务对应的.wav音频文件，或是praat所支持的文件格式，或是包含wav文件的文件夹
        :param input_f_audios_sig_ddk: 输入单任务DDK任务对应的.wav音频文件，或是praat所支持的文件格式，或是包含wav文件的文件夹
        :param input_f_rec: 输入测试的记录csv文件
        :param dt_num: 双任务编号
        :param f0min: 最小追踪pitch,默认75Hz
        :param f0max: 最大追踪pitch,默认600Hz
        :param sil_thr: 相对于音频最大强度的最大静音强度值(dB)。如imax是最大强度，则最大静音强度计算为sil_db=imax-|sil_thr|
                        强度小于sil_db的间隔被认为是静音段。
                        sil_thr越大，则语音段越可能被识别为静音段，这里默认为-15dB，正常语句说话需要设置小一些
        :param min_sil: 被认为是静音段的最小持续时间(s)。
                        默认0.005s，即该值越大，则语音段越可能被识别为有声段（若想把爆破音归为有声段，则将该值设置较大）
        :param min_snd: 被认为是有声段的最小持续时间(s)，即不被视为静音段的最小持续时间。
                        默认0.03s，低于该值被认为是静音段，即该值越大，则语音段越可能被识别为静音段
        """
        input_f_audios_dt_ddk = os.path.join(input_f_audios_dt_ddk, f"task{dt_num}")
        self.ddk_sig = DDKFeatures(input_f_audios_sig_ddk, f0min, f0max, sil_thr, min_sil, min_snd)
        self.ddk_dt = DDKFeatures(input_f_audios_dt_ddk, f0min, f0max, f0max, sil_thr, min_sil, min_snd)
        self.f_rec = input_f_rec
        self.dt_num = dt_num

    def dtc_ddk_rate(self):
        """
        计算双任务成本：DDK任务速率成本，(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.ddk_dt.ddk_rate() - self.ddk_sig.ddk_rate()) / self.ddk_sig.ddk_rate() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_ddk_regularity(self):
        """
        计算双任务成本：DDK任务规律性成本
        :return: DTC, %
        """
        try:
            return (self.ddk_dt.ddk_regularity() - self.ddk_sig.ddk_regularity()) / self.ddk_sig.ddk_regularity() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_ddk_duration_mean(self):
        """
        计算双任务成本：DDK任务中浊音段时长均值成本
        :return: DTC, %
        """
        try:
            return (self.ddk_dt.ddk_duration_mean() - self.ddk_sig.ddk_duration_mean()) / self.ddk_sig.ddk_duration_mean() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_voice_onset_time(self, diff_time_factor=0.5, diff_inten_factor=0.5):
        """
        计算双任务成本：DDK任务中个全部3个音节辅音p/t/k的平均时长成本
        :param diff_time_factor: 基频周期的diff_time_factor倍作为VOT时间阈值, 该值越大，限制越严格,默认0.5
        :param diff_inten_factor: 强度极值差的diff_inten_factor倍作为VOT对应的强度阈值, 该值越大，限制越严格,默认0.5
        :return: DTC, %
        """
        try:
            return (self.ddk_dt.voice_onset_time(diff_time_factor, diff_inten_factor)[0] -
                    self.ddk_sig.voice_onset_time(diff_time_factor, diff_inten_factor)[0]) / \
                   self.ddk_sig.voice_onset_time(diff_time_factor, diff_inten_factor)[0] * 100
        except ZeroDivisionError:
            return np.nan

    def _dtc_pause_rate(self):
        """
        计算双任务成本：DDK任务中停顿的速率成本
        :return: DTC, %
        """
        try:
            return (self.ddk_dt._pause_rate() - self.ddk_sig._pause_rate()) / self.ddk_sig._pause_rate() * 100
        except ZeroDivisionError:
            return np.nan

    def _dtc_pause_regularity(self):
        """
        计算双任务成本：DDK任务中停顿的规律性成本
        :return: DTC, %
        """
        try:
            return (self.ddk_dt._pause_regularity() - self.ddk_sig._pause_regularity()) / self.ddk_sig._pause_regularity() * 100
        except ZeroDivisionError:
            return np.nan

    def _dtc_pause_duration_mean(self):
        """
        计算双任务成本：DDK任务中停顿段时长均值成本
        :return: DTC, %
        """
        try:
            return (self.ddk_dt._pause_duration_mean() - self.ddk_sig._pause_duration_mean()) / self.ddk_sig._pause_duration_mean() * 100
        except ZeroDivisionError:
            return np.nan

    def wm_correct_num(self):
        """
        双任务中的次任务：工作记忆任务正确数
        :return: 工作记忆任务正确数
        """
        loc = f"DDK-视觉工作记忆双任务(Task{self.dt_num}_"
        cor_num = 0
        for i_row in read_csv(self.f_rec):
            if i_row[0].startswith(loc):
                cor_num += int(i_row[-1])
        return cor_num

    def get_all_feat(self, diff_time_factor=0.5, diff_inten_factor=0.5, prefix='DDKWM_', use_=False):
        """
        获取当前所有特征
        :param diff_time_factor: 基频周期的diff_time_factor倍作为VOT时间阈值, 该值越大，限制越严格,默认0.5
        :param diff_inten_factor: 强度极值差的diff_inten_factor倍作为VOT对应的强度阈值, 该值越大，限制越严格,默认0.5
        :param prefix: pd.DataFrame类型特征列名的前缀，默认没有
        :param use_: 是否提取包含分类贡献度低、ICC低的（方法有_标记）特征，默认否
        :return: 该类的全部特征, pd.DataFrame类型
        """
        dtc_ddk_rate = self.dtc_ddk_rate()
        dtc_ddk_regularity = self.dtc_ddk_regularity()
        dtc_ddk_duration_mean = self.dtc_ddk_duration_mean()
        dtc_vot_mean = self.dtc_voice_onset_time(diff_time_factor, diff_inten_factor)
        wm_cor_num = self.wm_correct_num()
        if use_:
            dtc_pause_rate = self._dtc_pause_rate()
            dtc_pause_regularity = self._dtc_pause_regularity()
            dtc_pause_duration_mean = self._dtc_pause_duration_mean()
            feat = {prefix + "DTC DDK rate(%)": [dtc_ddk_rate],
                    prefix + "DTC DDK regularity(%)": [dtc_ddk_regularity],
                    prefix + "DTC DDK duration(%)": [dtc_ddk_duration_mean],
                    prefix + "DTC VOT(%)": [dtc_vot_mean],
                    prefix + "DTC pause rate(%)": [dtc_pause_rate],
                    prefix + "DTC pause regularity(%)": [dtc_pause_regularity],
                    prefix + "DTC pause duration(%)": [dtc_pause_duration_mean],
                    prefix + "WM Correct Count": [wm_cor_num], }
        else:
            feat = {prefix + "DTC DDK rate(%)": [dtc_ddk_rate],
                    prefix + "DTC DDK regularity(%)": [dtc_ddk_regularity],
                    prefix + "DTC DDK duration(%)": [dtc_ddk_duration_mean],
                    prefix + "DTC VOT(%)": [dtc_vot_mean],
                    prefix + "WM Correct Count": [wm_cor_num], }
        return pd.DataFrame(feat)


if __name__ == "__main__":
    sid = os.path.basename(DATA_PATH_EXAMPLE).split('_')[-1]
    audio_files_dt_ddk = os.path.join(DATA_PATH_EXAMPLE, f"session_1/13_DDKDT")
    audio_files_sig_ddk = os.path.join(DATA_PATH_EXAMPLE, f"session_1/02_DDK")
    record_file = os.path.join(DATA_PATH_EXAMPLE, f"session_1/{sid}.csv")
    ddkwm_feat = DDKWMFeatures(audio_files_dt_ddk, audio_files_sig_ddk, record_file, 1).get_all_feat()
    print(ddkwm_feat)
