#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/2/26 16:32
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : 01_SP.py
# @Software : Python3.6; PyCharm; Windows10
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M
# @Version  : V1.3.0 - ZL.Z：2022/5/19
# 		      使用To TextGrid (silences)方法替换To TextGrid (vuv)，仅分割有声和无声段
#             V1.2.1 - ZL.Z：2022/5/5
# 		      输入支持文件夹，当为文件夹输入时，计算的特征为该文件夹下所有对应音频的特征均值
#             V1.2 - ZL.Z：2022/3/13
# 		      根据分析结果，将一些相关性高、分类贡献度低、ICC低等的特征做标记隐藏处理（方法前加_标记）
#             V1.0 - ZL.Z：2022/2/26
# 		      First version.
# @License  : None
# @Brief    : 提取持续元音发音任务的声学特征: jitter(%)/shimmer(%)/HNR(dB)/F0 mean(Hz)/F0 sd(st)/PPQ/APQ/DUV(%)/
#             formant1(Hz)/formant2(Hz)/MPT(s)/CPP(dB)

from src.utils.util import *
from src.config import *
import numpy as np
import re
import glob


class SPFeatures:
    """获取SP任务的发音特征"""
    def __init__(self, input_f_audios, f0min=75, f0max=600):
        """
        初始化
        :param input_f_audios: 输入.wav音频文件，或是praat所支持的文件格式，或是包含wav文件的文件夹
        :param f0min: 最小追踪pitch,默认75Hz
        :param f0max: 最大追踪pitch,默认600Hz
        """
        if os.path.isfile(input_f_audios):
            self.f_audio_l = [input_f_audios]
        else:
            self.f_audio_l = glob.glob(os.path.join(input_f_audios, r'*.wav'))
        self.f0min = f0min
        self.f0max = f0max
        self.total_duration_l, self.sound_l = self.func_total_time_use()
        self.point_process_l = []
        for i_sd in self.sound_l:
            self.point_process_l.append(call(i_sd, "To PointProcess (periodic, cc)", self.f0min, self.f0max))

    def func_total_time_use(self):
        """
        计算去掉头尾静音段之后的总用时以及对应的音频段
        :return: 总用时，单位s；对应的音频，parselmouth.Sound类型，这里都为列表，列表元素对应为各自的音频
        """
        time_len_l, sound_l = [], []
        for i_ad in self.f_audio_l:
            sd_obj = parselmouth.Sound(i_ad)
            text_grid_vuv = call(sd_obj, "To TextGrid (silences)", 100, 0.0, -25.0, 0.1, 0.1, 'U', 'V')
            vuv_info = call(text_grid_vuv, "List", 'no', 6, 'no', 'no')
            segments_v, segments_u = duration_from_vuvInfo(vuv_info)
            if len(segments_v):  # 为浊音
                time_len_l.append(segments_v[-1][-1] - segments_v[0][0])
                sound_l.append(call(sd_obj, 'Extract part', segments_v[0][0], segments_v[-1][-1],
                                    'rectangular', 1.0, 'no'))
            else:  # 为清音
                time_len_l.append(segments_u[-1][-1] - segments_u[0][0])
                sound_l.append(sd_obj)
        return time_len_l, sound_l

    def _jitter_local(self):
        """
        获取本地频率微扰jitter，单位为 %
        :return: jitt，float,单位为 %
        """
        jitt_l = []
        for i_pp in self.point_process_l:
            jitt_l.append(call(i_pp, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3) * 100)
        return np.mean(jitt_l)

    def _shimmer_local(self):
        """
        获取本地振幅微扰shimmer，单位为 %
        :return: shim，float,单位为 %
        """
        shim_l = []
        for i_ind in range(len(self.f_audio_l)):
            shim_l.append(call([self.sound_l[i_ind], self.point_process_l[i_ind]],
                               "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6) * 100)
        return np.mean(shim_l)

    def harmonic_to_noise_ratio(self):
        """
        获取谐噪比HNR
        :return: 谐噪比hnr,float, dB
        """
        hnr_l = []
        for i_sd in self.sound_l:
            harmonicity = call(i_sd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr_l.append(call(harmonicity, "Get mean", 0.0, 0.0))
        return np.mean(hnr_l)

    def f0_mean(self, unit="Hertz"):
        """
        获取基频pitch
        :param unit: 单位, Hertz(默认)/Hertz(logarithmic)/mel/logHertz/semitones re 1Hz/semitones re 100Hz/
                        semitones re 200Hz/semitones re 440 Hz/ERB
        :return: f0, float, Hz(unit="Hertz")
        """
        f0_l = []
        for i_sd in self.sound_l:
            pitch = call(i_sd, "To Pitch", 0.0, self.f0min, self.f0max)  # create a praat pitch object
            f0_l.append(call(pitch, "Get mean", 0, 0, unit))  # get mean pitch
        return np.mean(f0_l)

    def _f0_sd_semi(self):
        """
        获取semitones尺度的基频pitch的标准差
        :return: f0, float, Semitones
        """
        f0_sd_l = []
        for i_sd in self.sound_l:
            pitch = call(i_sd, "To Pitch", 0.0, self.f0min, self.f0max)  # create a praat pitch object
            f0_sd_l.append(call(pitch, "Get standard deviation", 0, 0, "semitones"))  # get standard deviation pitch
        return np.mean(f0_sd_l)

    def ppq(self):
        """
        获取jitter（PPQ5） Period perturbation quotient (PPQ)
        :return: ppq5, float
        """
        ppq5_l = []
        for i_pp in self.point_process_l:
            ppq5_l.append(call(i_pp, "Get jitter (ppq5)", 0.0, 0.0, 0.0001, 0.02, 1.3))
        return np.mean(ppq5_l)

    def apq(self):
        """
        获取shimmer（APQ11） Amplitude perturbation quotient (APQ)
        :return: apq11, float
        """
        apq11_l = []
        for i_ind in range(len(self.f_audio_l)):
            apq11_l.append(call([self.sound_l[i_ind], self.point_process_l[i_ind]],
                                "Get shimmer (apq11)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6))
        return np.mean(apq11_l)

    def _duv(self):
        """
        获取Fraction of locally unvoiced pitch frames,即持续元音发音中清音所占比例
        degree U(%) = Total duration of unvoiced frames / Total duration of the utterance * 100
        :return: degreeU, float, %
        """
        degreeU_l = []
        for i_ind in range(len(self.f_audio_l)):
            pitch = call(self.sound_l[i_ind], "To Pitch", 0.0, self.f0min, self.f0max)
            voice_report_str = call([self.sound_l[i_ind], pitch, self.point_process_l[i_ind]], "Voice report",
                                    0.0, 0.0, self.f0min, self.f0max, 1.3, 1.6, 0.03, 0.45)
            try:
                degreeU_l.append(float(re.findall("Fraction of locally unvoiced frames: (.*)%", voice_report_str)[0]))
            except IndexError:
                degreeU_l.append(0)
        return np.mean(degreeU_l)

    def formant(self, formant_index=1, formant_num=5):
        """
        获取第_i个共振峰的中心频率均值
        :param formant_index: 共振峰索引
        :param formant_num: 共振峰数量
        :return: formant_i, float, Hz
        """
        if formant_index > formant_num:
            raise ValueError("共振峰数量小于索引，请增大数量formant_num")
        formant_i_l = []
        for i_sd in self.sound_l:
            formants = call(i_sd, "To Formant (burg)", 0, formant_num, 5500, 0.025, 50.0)
            formant_i_l.append(call(formants, "Get mean", formant_index, 0.0, 0.0, "hertz"))
        return np.mean(formant_i_l)

    def mpt(self):
        """
        根据语音分割，获取语音段实际的发音时长
        :return: 实际的发音时长，float，单位s
        """
        return np.mean(self.total_duration_l)

    def cpp(self):
        """
        计算语音的倒谱峰突出 cepstral peak prominence
        :return: cpp，float，单位dB
        """
        cpp_l = []
        for i_sd in self.sound_l:
            power_cepstrogram = call(i_sd, "To PowerCepstrogram", 60.0, 0.002, 8000.0, 50.0)
            cpp_l.append(call(power_cepstrogram, "Get CPPS", True, 0.02, 0.0005, 60.0, 330.0, 0.05,
                              "parabolic", 0.001, 0.05, "Exponential decay", "Robust"))  # slow
        return np.mean(cpp_l)

    def get_all_feat(self, prefix='', use_=False):
        """
        获取当前所有特征
        :param prefix: pd.DataFrame类型特征列名的前缀，默认没有
        :param use_: 是否提取包含相关性高、分类贡献度低、ICC低等的（方法有_标记）特征，默认否
        :return: 该类的全部特征, pd.DataFrame类型
        """
        hnr = self.harmonic_to_noise_ratio()
        f0_mean = self.f0_mean()
        ppq = self.ppq()
        apq = self.apq()
        formant1 = self.formant(1)
        formant2 = self.formant(2)
        mpt = self.mpt()
        cpp = self.cpp()
        if use_:
            jitt = self._jitter_local()
            shimm = self._shimmer_local()
            f0_sd = self._f0_sd_semi()
            duv = self._duv()
            feat = {prefix+"Jitter(%)": [jitt], prefix+"Shimmer(%)": [shimm], prefix+"HNR(dB)": [hnr],
                    prefix+"F0(Hz)": [f0_mean], prefix+"F0 SD(st)": [f0_sd],
                    prefix+"PPQ": [ppq], prefix+"APQ": [apq], prefix+"DUV(%)": [duv],
                    prefix+"Formant1(Hz)": [formant1], prefix+"Formant2(Hz)": [formant2],
                    prefix+"MPT(s)": [mpt], prefix+"CPP(dB)": [cpp]}
        else:
            feat = {prefix + "HNR(dB)": [hnr], prefix + "F0(Hz)": [f0_mean], prefix + "PPQ": [ppq],
                    prefix + "APQ": [apq], prefix + "Formant1(Hz)": [formant1], prefix + "Formant2(Hz)": [formant2],
                    prefix + "MPT(s)": [mpt], prefix + "CPP(dB)": [cpp]}
        return pd.DataFrame(feat)


if __name__ == "__main__":
    sid = os.path.basename(DATA_PATH_EXAMPLE).split('_')[-1]
    audio_file = os.path.join(DATA_PATH_EXAMPLE, f"session_1/01_SP/{sid}_sp1.wav")
    audio_files = os.path.join(DATA_PATH_EXAMPLE, f"session_1/01_SP")
    sp_feat = SPFeatures(audio_files).get_all_feat()
    print(sp_feat)



