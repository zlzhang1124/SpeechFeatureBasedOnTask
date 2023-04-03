#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2021/5/19 08:30
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : DDK.py
# @Software : Python3.6; PyCharm; Windows10
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M
# @Version  : V1.3.0 - ZL.Z：2022/5/19
# 		      使用To TextGrid (silences)方法替换To TextGrid (vuv)，仅分割有声和无声段，提高该特征准确性
#             V1.2.2 - ZL.Z：2022/5/5
# 		      输入支持文件夹，当为文件夹输入时，计算的特征为该文件夹下所有对应音频的特征均值
#             V1.2.1 - ZL.Z：2022/4/8
# 		      添加VOT中分割的音频波形可视化
#             V1.2 - ZL.Z：2022/3/13
# 		      根据分析结果，将一些分类贡献度低、ICC低的特征做标记隐藏处理（方法前加_标记）
#             V1.1 - ZL.Z：2022/2/26
# 		      vup_duration_from_vuvTextGrid改为duration_from_vuvInfo
#             V1.0 - ZL.Z：2021/5/19
# 		      First version.
# @License  : None
# @Brief    : 提取DDK任务的声学特征: DDK rate(syll/s)/DDK regularity(ms)/DDK mean duration(ms)/mean VOT(ms)
#                                pause rate(1/s)/pause regularity(ms)/pause mean duration(ms)

from src.utils.util import *
from src.config import *
import numpy as np
from scipy.signal import argrelextrema
import glob


class DDKFeatures:
    """获取DDK任务的发音特征"""
    def __init__(self, input_f_audios, f0min=75, f0max=600, sil_thr=-15.0, min_sil=0.005, min_snd=0.03):
        """
        初始化
        :param input_f_audios: 输入.wav音频文件，或是praat所支持的文件格式，或是包含wav文件的文件夹
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
        if os.path.isfile(input_f_audios):
            self.f_audio_l = [input_f_audios]
        else:
            self.f_audio_l = glob.glob(os.path.join(input_f_audios, r'*.wav'))
        self.f0min = f0min
        self.f0max = f0max
        self.sound_l, self.point_process_l, self.text_grid_vuv_l, \
        self.vuv_info_l, self.total_duration_l = [], [], [], [], []
        for i_ad in self.f_audio_l:
            sound = parselmouth.Sound(i_ad)
            sound.scale_peak(0.99)
            self.sound_l.append(sound)
            self.text_grid_vuv_l.append(call(sound, "To TextGrid (silences)", 100, 0.0,
                                             sil_thr, min_sil, min_snd, 'U', 'V'))
            self.vuv_info_l.append(call(self.text_grid_vuv_l[-1], "List", False, 10, False, False))
            self.total_duration_l.append(sound.get_total_duration())

    def ddk_rate(self):
        """
        计算DDK任务速率：每秒发音的音节数
        :return: rate, float, syll/s
        """
        rate_l = []
        for i_ind in range(len(self.f_audio_l)):
            segments_v, segments_u = duration_from_vuvInfo(self.vuv_info_l[i_ind])
            rate_l.append(len(segments_v) / self.total_duration_l[i_ind])
        return np.mean(rate_l)

    def ddk_regularity(self):
        """
        计算DDK任务规律性：音节发音间距离（持续时间）的标准差
        :return: regularity, float, ms
        """
        regularity_l = []
        for i_vi in self.vuv_info_l:
            segments_v, segments_u = duration_from_vuvInfo(i_vi)
            if len(segments_v) == 0:  # 空列表表示该音频没有浊音段，此时直接返回0.0
                return 0.0
            duration_list = []
            for duration in segments_v:
                duration_list.append(duration[1] - duration[0])
            regularity_l.append(1000 * np.std(duration_list))
        return np.mean(regularity_l)

    def ddk_duration_mean(self):
        """
        计算DDK任务中浊音段时长均值
        :return: duration_mean, float, ms
        """
        duration_mean_l = []
        for i_vi in self.vuv_info_l:
            segments_v, segments_u = duration_from_vuvInfo(i_vi)
            if len(segments_v) == 0:  # 空列表表示该音频没有浊音段，此时直接返回0.0
                return 0.0
            duration_list = []
            for duration in segments_v:
                duration_list.append(duration[1] - duration[0])
            duration_mean_l.append(1000 * np.mean(duration_list))
        return np.mean(duration_mean_l)

    def voice_onset_time(self, diff_time_factor=0.5, diff_inten_factor=0.5, pt=False):
        """
        计算DDK任务中个全部3个音节辅音p/t/k的平均时长：voice onset time，这里采用音频包络(强度曲线intensity)的极值法
        :param diff_time_factor: 基频周期的diff_time_factor倍作为VOT时间阈值, 该值越大，限制越严格,默认0.5
        :param diff_inten_factor: 强度极值差的diff_inten_factor倍作为VOT对应的强度阈值, 该值越大，限制越严格,默认0.5
        :param pt: 是否绘制每个音节的initial burst和vowel onset竖线，以及峰值极值点，默认False
        :return: vot_mean(float, ms), [{(initial burst time, vowel onset time): vot}(元祖，单位都为ms)]
                 其中， vot = vowel onset time - initial burst time
        """
        vot_mean_l, vot_dict_l = [], []
        for i_ind in range(len(self.f_audio_l)):
            intensity_obj = call(self.sound_l[i_ind], "To Intensity", 100, 0.0)
            intensity_matrix_obj = call(intensity_obj, "Down to Matrix")
            intensity_list = call(intensity_matrix_obj, "Get all values")
            # 音频强度intensity包络极值点
            intensity_list_greater = intensity_list[0][argrelextrema(intensity_list[0], np.greater)[0]].tolist()
            intensity_list_less = intensity_list[0][argrelextrema(intensity_list[0], np.less)[0]].tolist()
            pitch_obj = call(self.sound_l[i_ind], "To Pitch", 0.0, self.f0min, self.f0max)
            f0_mean = call(pitch_obj, "Get mean", 0.0, 0.0, "Hertz")
            intensity_mean = call(intensity_obj, "Get mean", 0.0, 0.0, "energy")
            diff_time_thr = diff_time_factor * 1 / f0_mean  # 基频周期的diff_time_factor倍作为VOT时间阈值
            diff_intensity_thr = diff_inten_factor * (np.mean(intensity_list_greater) -
                                                      np.mean(intensity_list_less))  # 强度极值差的diff_inten_factor倍作为VOT对应的强度阈值
            initial_burst_dict = {}
            vowel_onset_dict = {}
            for t in range(len(intensity_list[0])):  # 获取每个极值点对应的的时间及其强度值，组合为元祖
                if len(intensity_list_less) and intensity_list[0][t] == intensity_list_less[0]:
                    initial_burst_dict[t * (self.total_duration_l[i_ind] / len(intensity_list[0]))] = intensity_list[0][t]
                    del(intensity_list_less[0])
                    continue
                elif len(intensity_list_greater) and intensity_list[0][t] == intensity_list_greater[0]:
                    vowel_onset_dict[t * (self.total_duration_l[i_ind] / len(intensity_list[0]))] = intensity_list[0][t]
                    del(intensity_list_greater[0])
                    continue
                else:
                    continue
            vot_dict = {}
            length = max(len(initial_burst_dict), len(vowel_onset_dict))
            for ind in range(length):
                try:
                    if list(vowel_onset_dict.values())[ind] < 0.8 * intensity_mean:  # 元音开始对应的强度值需满足的条件
                        continue
                    diff_time = list(vowel_onset_dict.keys())[ind] - list(initial_burst_dict.keys())[ind]
                    if diff_time > 0:  # 时间差值为正，说明极大值（元音开始）数量大于等于极小值（爆破音开始）数量
                        if diff_time < diff_time_thr:  # 需满足VOT时间阈值条件
                            continue
                        diff_intensity = list(vowel_onset_dict.values())[ind] - list(initial_burst_dict.values())[ind]
                        if diff_intensity < diff_intensity_thr:  # 需满足VOT对应的强度值阈值条件
                            continue
                        # 根据时间对应关系，找到相邻两极值点对应强度值的中点，其对应的时间即作为元音开始的时间
                        intensity_middle = list(initial_burst_dict.values())[ind] + diff_intensity / 2.0
                        idx_down = round(list(initial_burst_dict.keys())[ind] / (self.total_duration_l[i_ind] / len(intensity_list[0])))
                        idx_up = round(list(vowel_onset_dict.keys())[ind] / (self.total_duration_l[i_ind] / len(intensity_list[0])))
                    else:
                        diff_time = list(vowel_onset_dict.keys())[ind + 1] - list(initial_burst_dict.keys())[ind]
                        if diff_time < diff_time_thr:
                            continue
                        diff_intensity = list(vowel_onset_dict.values())[ind + 1] - list(initial_burst_dict.values())[ind]
                        if diff_intensity < diff_intensity_thr:
                            continue
                        intensity_middle = list(initial_burst_dict.values())[ind] + diff_intensity / 2.0
                        idx_down = round(
                            list(initial_burst_dict.keys())[ind] / (self.total_duration_l[i_ind] / len(intensity_list[0])))
                        idx_up = round(list(vowel_onset_dict.keys())[ind+1] / (self.total_duration_l[i_ind] / len(intensity_list[0])))
                    # 该强度中点在原始采样点中不一定存在，此时找到最接近的数值记为该中点值
                    idx = idx_down + (
                        np.abs(np.asarray(list(intensity_list[0])[idx_down:idx_up + 1]) - intensity_middle)).argmin()
                    time_middle = idx * (self.total_duration_l[i_ind] / len(intensity_list[0]))
                    if 1000*diff_time < 18:  # 防止噪音抖动，若间隔值太小，则爆破音初始往左移动一步（爆破和元音强度值相差不大时）
                        try:
                            if abs(list(initial_burst_dict.values())[ind] -
                                   list(vowel_onset_dict.values())[ind-1]) < 3.5:
                                vot_dict[(1000 * list(initial_burst_dict.keys())[ind-1], 1000 * time_middle)] = \
                                    1000 * (time_middle - list(initial_burst_dict.keys())[ind])
                            else:
                                pass
                        except IndexError:
                            pass
                    else:
                        vot_dict[(1000*list(initial_burst_dict.keys())[ind], 1000*time_middle)] = \
                            1000*(time_middle - list(initial_burst_dict.keys())[ind])
                except IndexError:
                    pass
            for key in list(vot_dict.keys()):  # 删除VOT小于15ms的片段（经验VOT>15ms）
                if vot_dict[key] < 15.0:
                    del vot_dict[key]
            if len(vot_dict) == 0:  # 若没检测到VOT，则提示重新设置参数
                import warnings
                warnings.warn("diff_inten_factor={}太大，此时检测不到VOT，请重新设置参数".format(diff_inten_factor))
                return 0.0, {}
            vot_mean_l.append(float(np.mean(list(vot_dict.values()))))
            vot_dict_l.append(vot_dict)
            if pt:  # 绘制每个音节的initial burst和vowel onset竖线，以及峰值极值点
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6), dpi=300)
                x = [i * (self.total_duration_l[i_ind] / len(intensity_list[0])) for i in range(len(intensity_list[0]))]
                plt.plot(x, intensity_list[0])
                for j in np.array(list(vot_dict.keys()))[:, 0]:
                    plt.axvline(x=j / 1000, c='g', ls="--", lw=0.6)
                for j in np.array(list(vot_dict.keys()))[:, 1]:
                    plt.axvline(x=j / 1000, c='r', ls="--", lw=0.6)
                plt.scatter(list(vowel_onset_dict.keys()), list(vowel_onset_dict.values()), c='red', s=15)
                plt.scatter(list(initial_burst_dict.keys()), list(initial_burst_dict.values()), c='g', s=15, marker='*')
                plt.twinx()
                sound_obj = call(self.sound_l[i_ind], "Down to Matrix")
                sound_list = call(sound_obj, "Get all values")
                plt.plot([i * (self.total_duration_l[i_ind] / len(sound_list[0])) for i in range(len(sound_list[0]))],
                         sound_list[0], c='gray')
                plt.show()
        return np.mean(vot_mean_l), vot_dict_l

    def _pause_rate(self):
        """
        计算DDK任务中停顿的速率：每秒停顿的次数
        :return: rate, float, 1/s
        """
        rate_l = []
        for i_ind in range(len(self.f_audio_l)):
            segments_v, segments_u = duration_from_vuvInfo(self.vuv_info_l[i_ind])
            rate_l.append(len(segments_u) / self.total_duration_l[i_ind])
        return np.mean(rate_l)

    def _pause_regularity(self):
        """
        计算DDK任务中停顿的规律性：发音中停顿持续时间的标准差
        :return: regularity, float, ms
        """
        regularity_l = []
        for i_vi in self.vuv_info_l:
            segments_v, segments_u = duration_from_vuvInfo(i_vi)
            if len(segments_u) == 0:  # 空列表表示该音频没有停顿段，此时直接返回0.0
                return 0.0
            duration_list = []
            for duration in segments_u:
                duration_list.append(duration[1] - duration[0])
            regularity_l.append(1000 * np.std(duration_list))
        return np.mean(regularity_l)

    def _pause_duration_mean(self):
        """
        计算DDK任务中停顿段时长均值
        :return: duration_mean, float, ms
        """
        duration_mean_l = []
        for i_vi in self.vuv_info_l:
            segments_v, segments_u = duration_from_vuvInfo(i_vi)
            if len(segments_u) == 0:  # 空列表表示该音频没有停顿段，此时直接返回0.0
                return 0.0
            duration_list = []
            for duration in segments_u:
                duration_list.append(duration[1] - duration[0])
            duration_mean_l.append(1000 * np.mean(duration_list))
        return np.mean(duration_mean_l)

    def get_all_feat(self, diff_time_factor=0.5, diff_inten_factor=0.5, pt=False, prefix='', use_=False):
        """
        获取当前所有特征
        :param diff_time_factor: 基频周期的diff_time_factor倍作为VOT时间阈值, 该值越大，限制越严格,默认0.5
        :param diff_inten_factor: 强度极值差的diff_inten_factor倍作为VOT对应的强度阈值, 该值越大，限制越严格,默认0.5
        :param pt: 是否绘制每个音节的initial burst和vowel onset竖线，以及峰值极值点，默认False
        :param prefix: pd.DataFrame类型特征列名的前缀，默认没有
        :param use_: 是否提取包含分类贡献度低、ICC低的（方法有_标记）特征，默认否
        :return: 该类的全部特征, pd.DataFrame类型
        """
        ddk_rate = self.ddk_rate()
        ddk_regularity = self.ddk_regularity()
        ddk_duration_mean = self.ddk_duration_mean()
        vot_mean, __ = self.voice_onset_time(diff_time_factor, diff_inten_factor, pt)
        if use_:
            pause_rate = self._pause_rate()
            pause_regularity = self._pause_regularity()
            pause_duration_mean = self._pause_duration_mean()
            feat = {prefix+"DDK rate(syll/s)": [ddk_rate], prefix+"DDK regularity(ms)": [ddk_regularity],
                    prefix+"DDK duration(ms)": [ddk_duration_mean], prefix+"VOT(ms)": [vot_mean],
                    prefix+"pause rate(1/s)": [pause_rate], prefix+"pause regularity(ms)": [pause_regularity],
                    prefix+"pause duration(ms)": [pause_duration_mean]}
        else:
            feat = {prefix + "DDK rate(syll/s)": [ddk_rate], prefix + "DDK regularity(ms)": [ddk_regularity],
                    prefix + "DDK duration(ms)": [ddk_duration_mean], prefix + "VOT(ms)": [vot_mean]}
        return pd.DataFrame(feat)


if __name__ == "__main__":
    sid = os.path.basename(DATA_PATH_EXAMPLE).split('_')[-1]
    audio_file = os.path.join(DATA_PATH_EXAMPLE, f"session_1/02_DDK/{sid}_ddk1.wav")
    audio_files = os.path.join(DATA_PATH_EXAMPLE, f"session_1/02_DDK")
    ddk_feat = DDKFeatures(audio_files).get_all_feat(use_=True)
    print(ddk_feat)

