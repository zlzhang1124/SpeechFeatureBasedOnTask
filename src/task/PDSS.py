#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/4/24 9:27
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : PDSS.py
# @Software : Python3.6; PyCharm; Windows10
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M
# @Version  : V1.1 - ZL.Z：2022/5/19
# 		      跟随SI.py修改
#             V1.0 - ZL.Z：2022/4/24
# 		      First version.
# @License  : None
# @Brief    : 提取图片描述（主）-连续减法（次）PD-SS双任务的特征: 以下特征对应的双任务成本
#                基于声学（语音和音律Phonetic and Phonological）：
#                F0 SD(st)/Intensity SD(dB)/DPI(ms)/RST(-/s)/EST/Voiced Rate(1/s)/Hesitation Ratio/
#                Energy Mean(Pa^2·s)/MFCC(39维)
#                基于文本：
#                       语义（词汇语义Lexico-semantic）：
#                       Word Num/Word Rate(-/s)/Function Word Ratio/Lexical Density/MATTR/Sentence Num/
#                       形态句法和句法（Morpho-syntactic & Syntactic）：MLU/Noun Phrase Rate/Verb Phrase Rate/
#                       Adj Phrase Rate/Adv Phrase Rate/Prep Phrase Rate/Parse Tree Height/Yngve Depth Max/
#                       Yngve Depth Mean/Yngve Depth Total/Frazier Depth Max/Frazier Depth Mean/Frazier Depth Total/
#                       Dependency Distance Total/Dependency Distance Mean
#                       语篇和语用（Discourse & Pragmatic）：Discourse Total/Discourse Rate/Information Units/Efficiency
#                       词向量：TF-IDF
#               次任务500连续减3任务的特征: Accuracy12/Correct Count

from src.task.PD import PDFeatures
from src.task.SS import SSFeatures
from src.config import *
import numpy as np


class PDSSFeatures:
    """获取PDSS任务的特征: 基于双任务成本的特征，仅在单任务PD任务存在时才能计算"""
    def __init__(self, input_f_audio_dt_pd: str, input_f_trans_dt_pd: str,
                 input_f_audio_dt_ss: str, input_f_trans_dt_ss: str,
                 input_f_audio_sig_pd: str, input_f_trans_sig_pd: str, f0min=75, f0max=600,
                 sil_thr=-25.0, min_sil=0.1, min_snd=0.1):
        """
        初始化
        :param input_f_audio_dt_pd: 输入PDSS双任务中PD任务对应的.wav音频文件，或是praat所支持的文件格式
        :param input_f_trans_dt_pd: 输入PDSS双任务中PD任务对应的文本转录文件，.txt或.cha类似的文件格式
        :param input_f_audio_dt_ss: 输入PDSS双任务中SS任务对应的.wav音频文件，或是praat所支持的文件格式
        :param input_f_trans_dt_ss: 输入PDSS双任务中SS任务对应的文本转录文件，.txt或.cha类似的文件格式
        :param input_f_audio_sig_pd: 输入单任务PD任务对应的.wav音频文件，或是praat所支持的文件格式
        :param input_f_trans_sig_pd: 输入单任务PD任务对应的文本转录文件，.txt或.cha类似的文件格式
        :param f0min: 最小追踪pitch,默认75Hz
        :param f0max: 最大追踪pitch,默认600Hz
        :param sil_thr: 相对于音频最大强度的最大静音强度值(dB)。如imax是最大强度，则最大静音强度计算为sil_db=imax-|sil_thr|
                        强度小于sil_db的间隔被认为是静音段。sil_thr越大，则语音段越可能被识别为静音段，这里默认为-25dB
        :param min_sil: 被认为是静音段的最小持续时间(s)。
                        默认0.1s，即该值越大，则语音段越可能被识别为有声段（若想把爆破音归为有声段，则将该值设置较大）
        :param min_snd: 被认为是有声段的最小持续时间(s)，即不被视为静音段的最小持续时间。
                        默认0.1s，低于该值被认为是静音段，即该值越大，则语音段越可能被识别为静音段
        """
        self.pd_sig = PDFeatures(input_f_audio_sig_pd, input_f_trans_sig_pd, f0min, f0max, sil_thr, min_sil, min_snd)
        self.pd_dt = PDFeatures(input_f_audio_dt_pd, input_f_trans_dt_pd, f0min, f0max, sil_thr, min_sil, min_snd)
        self.ss_dt = SSFeatures(input_f_audio_dt_ss, input_f_trans_dt_ss, f0min, f0max)

    def dtc_f0_std(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        return (self.pd_dt.f0_std() - self.pd_sig.f0_std()) / self.pd_sig.f0_std() * 100

    def _dtc_intensity_std(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        return (self.pd_dt._intensity_std() - self.pd_sig._intensity_std()) / self.pd_sig._intensity_std() * 100

    def dtc_duration_pause_intervals(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.duration_pause_intervals() - self.pd_sig.duration_pause_intervals()) / self.pd_sig.duration_pause_intervals() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_rate_speech_timing(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.rate_speech_timing() - self.pd_sig.rate_speech_timing()) / self.pd_sig.rate_speech_timing() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_entropy_speech_timing(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.entropy_speech_timing() - self.pd_sig.entropy_speech_timing()) / self.pd_sig.entropy_speech_timing() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_voiced_rate(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.voiced_rate() - self.pd_sig.voiced_rate()) / self.pd_sig.voiced_rate() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_hesitation_ratio(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.hesitation_ratio() - self.pd_sig.hesitation_ratio()) / self.pd_sig.hesitation_ratio() * 100
        except ZeroDivisionError:
            return np.nan

    def _dtc_energy_mean(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt._energy_mean() - self.pd_sig._energy_mean()) / self.pd_sig._energy_mean() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_word_num_all(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.word_num_all() - self.pd_sig.word_num_all()) / self.pd_sig.word_num_all() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_word_rate(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.word_rate() - self.pd_sig.word_rate()) / self.pd_sig.word_rate() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_function_word_ratio(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.function_word_ratio() - self.pd_sig.function_word_ratio()) / self.pd_sig.function_word_ratio() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_lexical_density(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.lexical_density() - self.pd_sig.lexical_density()) / self.pd_sig.lexical_density() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_lexical_diversity(self, win_len=50):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt._lexical_diversity(win_len) - self.pd_sig._lexical_diversity(win_len)) / \
                   self.pd_sig._lexical_diversity(win_len) * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_sentence_num(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.sentence_num() - self.pd_sig.sentence_num()) / self.pd_sig.sentence_num() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_mean_len_utter(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.mean_len_utter() - self.pd_sig.mean_len_utter()) / self.pd_sig.mean_len_utter() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_noun_phrase_rate(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.noun_phrase_rate() - self.pd_sig.noun_phrase_rate()) / self.pd_sig.noun_phrase_rate() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_verb_phrase_rate(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.verb_phrase_rate() - self.pd_sig.verb_phrase_rate()) / self.pd_sig.verb_phrase_rate() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_adj_phrase_rate(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.adj_phrase_rate() - self.pd_sig.adj_phrase_rate()) / self.pd_sig.adj_phrase_rate() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_adv_phrase_rate(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.adv_phrase_rate() - self.pd_sig.adv_phrase_rate()) / self.pd_sig.adv_phrase_rate() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_prep_phrase_rate(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.prep_phrase_rate() - self.pd_sig.prep_phrase_rate()) / self.pd_sig.prep_phrase_rate() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_parse_tree_height(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.parse_tree_height() - self.pd_sig.parse_tree_height()) / self.pd_sig.parse_tree_height() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_max_yngve_depth(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.max_yngve_depth() - self.pd_sig.max_yngve_depth()) / self.pd_sig.max_yngve_depth() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_mean_yngve_depth(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.mean_yngve_depth() - self.pd_sig.mean_yngve_depth()) / self.pd_sig.mean_yngve_depth() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_total_yngve_depth(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.total_yngve_depth() - self.pd_sig.total_yngve_depth()) / self.pd_sig.total_yngve_depth() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_max_frazier_depth(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.max_frazier_depth() - self.pd_sig.max_frazier_depth()) / self.pd_sig.max_frazier_depth() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_mean_frazier_depth(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.mean_frazier_depth() - self.pd_sig.mean_frazier_depth()) / self.pd_sig.mean_frazier_depth() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_total_frazier_depth(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.total_frazier_depth() - self.pd_sig.total_frazier_depth()) / self.pd_sig.total_frazier_depth() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_total_dependency_distance(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.total_dependency_distance() - self.pd_sig.total_dependency_distance()) / self.pd_sig.total_dependency_distance() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_mean_dependency_distance(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.mean_dependency_distance() - self.pd_sig.mean_dependency_distance()) / self.pd_sig.mean_dependency_distance() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_total_discourse_markers(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.total_discourse_markers() - self.pd_sig.total_discourse_markers()) / self.pd_sig.total_discourse_markers() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_discourse_marker_rate(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.discourse_marker_rate() - self.pd_sig.discourse_marker_rate()) / self.pd_sig.discourse_marker_rate() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_total_information_units_ner(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.total_information_units_ner() - self.pd_sig.total_information_units_ner()) / \
                   self.pd_sig.total_information_units_ner() * 100
        except ZeroDivisionError:
            return np.nan

    def dtc_efficiency_information_units_ner(self):
        """
        计算双任务范式下的双任务成本DTC：(双任务特征-单任务特征)/单任务特征*100
        :return: DTC, %
        """
        try:
            return (self.pd_dt.efficiency_information_units_ner() - self.pd_sig.efficiency_information_units_ner()) / \
                   self.pd_sig.efficiency_information_units_ner() * 100
        except ZeroDivisionError:
            return np.nan

    def mfcc(self):
        """
        计算39维MFCC系数：13个MFCC特征（第一个系数为能量c0）及其对应的一阶和二阶差分
        :return: 13*3维MFCC特征，每一列为一个MFCC特征向量 np.ndarray[shape=(n_mfcc*3, n_frames), dtype=float64]
        """
        return self.pd_dt.mfcc()

    def tf_idf(self, stop_words=None):
        """
        计算文本的词频-逆文档频率TF-IDF
        :param stop_words: 过滤的停用词列表，默认['是', '的', '了']
        :return: tf_idf, np.narray[(n_words(排除标点、重复、停用词),), float64]
        """
        return self.pd_dt.tf_idf(stop_words)

    def get_all_feat(self, prefix='PDSS_', use_=False):
        """
        获取当前所有特征
        :param prefix: pd.DataFrame类型特征列名的前缀
        :param use_: 是否提取包含分类贡献度低、ICC低的（方法有_标记）特征，默认否
        :return: 该类的全部特征, pd.DataFrame类型
        """
        dtc_f0_sd = self.dtc_f0_std()
        dtc_dpi = self.dtc_duration_pause_intervals()
        dtc_rst = self.dtc_rate_speech_timing()
        dtc_est = self.dtc_entropy_speech_timing()
        dtc_voiced_rate = self.dtc_voiced_rate()
        dtc_hesi_r = self.dtc_hesitation_ratio()
        mfcc = self.mfcc()
        dtc_wn = self.dtc_word_num_all()
        dtc_wr = self.dtc_word_rate()
        dtc_fwr = self.dtc_function_word_ratio()
        dtc_ld = self.dtc_lexical_density()
        dtc_mattr = self.dtc_lexical_diversity()
        dtc_sn = self.dtc_sentence_num()
        dtc_mlu = self.dtc_mean_len_utter()
        dtc_n_pr = self.dtc_noun_phrase_rate()
        dtc_v_pr = self.dtc_verb_phrase_rate()
        dtc_adj_pr = self.dtc_adj_phrase_rate()
        dtc_adv_pr = self.dtc_adv_phrase_rate()
        dtc_prep_pr = self.dtc_prep_phrase_rate()
        dtc_tree_h = self.dtc_parse_tree_height()
        dtc_yngve_max = self.dtc_max_yngve_depth()
        dtc_yngve_mean = self.dtc_mean_yngve_depth()
        dtc_yngve_total = self.dtc_total_yngve_depth()
        dtc_frazier_max = self.dtc_max_frazier_depth()
        dtc_frazier_mean = self.dtc_mean_frazier_depth()
        dtc_frazier_total = self.dtc_total_frazier_depth()
        dtc_dep_dist_total = self.dtc_total_dependency_distance()
        dtc_dep_dist_mean = self.dtc_mean_dependency_distance()
        dtc_disc_total = self.dtc_total_discourse_markers()
        dtc_disc_r = self.dtc_discourse_marker_rate()
        dtc_iu_ner_total = self.dtc_total_information_units_ner()
        dtc_iu_ner_eff = self.dtc_efficiency_information_units_ner()
        idf = self.tf_idf()
        acc = self.ss_dt.accuracy(start_number=500, difference_value=7, n_num=12)
        num_cor = self.ss_dt._total_num_correct(start_number=500, difference_value=7)
        if use_:
            dtc_int_sd = self._dtc_intensity_std()
            dtc_energy = self._dtc_energy_mean()
            feat = {prefix + "DTC F0 SD(%)": [dtc_f0_sd], prefix + "DTC Intensity SD(%)": [dtc_int_sd],
                    prefix + "DTC DPI(%)": [dtc_dpi], prefix + "DTC RST(%)": [dtc_rst], prefix + "DTC EST(%)": [dtc_est],
                    prefix + "DTC Voiced Rate(%)": [dtc_voiced_rate],
                    prefix + "DTC Hesitation Ratio(%)": [dtc_hesi_r], prefix + "DTC Energy Mean(%)": [dtc_energy],
                    prefix + "MFCC": [mfcc], prefix + "DTC Word Num(%)": [dtc_wn],
                    prefix + "DTC Word Rate(%)": [dtc_wr], prefix + "DTC Function Word Ratio(%)": [dtc_fwr],
                    prefix + "DTC Lexical Density(%)": [dtc_ld],
                    prefix + "DTC MATTR(%)": [dtc_mattr], prefix + "DTC Sentence Num(%)": [dtc_sn], prefix + "DTC MLU(%)": [dtc_mlu],
                    prefix + "DTC Noun Phrase Rate(%)": [dtc_n_pr], prefix + "DTC Verb Phrase Rate(%)": [dtc_v_pr],
                    prefix + "DTC Adj Phrase Rate(%)": [dtc_adj_pr], prefix + "DTC Adv Phrase Rate(%)": [dtc_adv_pr],
                    prefix + "DTC Prep Phrase Rate(%)": [dtc_prep_pr], prefix + "DTC Parse Tree Height(%)": [dtc_tree_h],
                    prefix + "DTC Yngve Depth Max(%)": [dtc_yngve_max], prefix + "DTC Yngve Depth Mean(%)": [dtc_yngve_mean],
                    prefix + "DTC Yngve Depth Total(%)": [dtc_yngve_total], prefix + "DTC Frazier Depth Max(%)": [dtc_frazier_max],
                    prefix + "DTC Frazier Depth Mean(%)": [dtc_frazier_mean], prefix + "DTC Frazier Depth Total(%)": [dtc_frazier_total],
                    prefix + "DTC Dependency Distance Total(%)": [dtc_dep_dist_total],
                    prefix + "DTC Dependency Distance Mean(%)": [dtc_dep_dist_mean],
                    prefix + "DTC Discourse Total(%)": [dtc_disc_total], prefix + "DTC Discourse Rate(%)": [dtc_disc_r],
                    prefix + "DTC Information Units(%)": [dtc_iu_ner_total], prefix + "DTC Efficiency(%)": [dtc_iu_ner_eff],
                    prefix + "TF-IDF": [idf], prefix + "Accuracy12": [acc],
                    prefix + "Correct Count": [num_cor], }
        else:
            feat = {prefix + "DTC F0 SD(%)": [dtc_f0_sd], prefix + "DTC DPI(%)": [dtc_dpi],
                    prefix + "DTC RST(%)": [dtc_rst], prefix + "DTC EST(%)": [dtc_est],
                    prefix + "DTC Voiced Rate(%)": [dtc_voiced_rate], prefix + "DTC Hesitation Ratio(%)": [dtc_hesi_r],
                    prefix + "MFCC": [mfcc], prefix + "DTC Word Num(%)": [dtc_wn],
                    prefix + "DTC Word Rate(%)": [dtc_wr], prefix + "DTC Function Word Ratio(%)": [dtc_fwr],
                    prefix + "DTC Lexical Density(%)": [dtc_ld], prefix + "DTC MATTR(%)": [dtc_mattr],
                    prefix + "DTC Sentence Num(%)": [dtc_sn], prefix + "DTC MLU(%)": [dtc_mlu],
                    prefix + "DTC Noun Phrase Rate(%)": [dtc_n_pr], prefix + "DTC Verb Phrase Rate(%)": [dtc_v_pr],
                    prefix + "DTC Adj Phrase Rate(%)": [dtc_adj_pr], prefix + "DTC Adv Phrase Rate(%)": [dtc_adv_pr],
                    prefix + "DTC Prep Phrase Rate(%)": [dtc_prep_pr], prefix + "DTC Parse Tree Height(%)": [dtc_tree_h],
                    prefix + "DTC Yngve Depth Max(%)": [dtc_yngve_max], prefix + "DTC Yngve Depth Mean(%)": [dtc_yngve_mean],
                    prefix + "DTC Yngve Depth Total(%)": [dtc_yngve_total], prefix + "DTC Frazier Depth Max(%)": [dtc_frazier_max],
                    prefix + "DTC Frazier Depth Mean(%)": [dtc_frazier_mean], prefix + "DTC Frazier Depth Total(%)": [dtc_frazier_total],
                    prefix + "DTC Dependency Distance Total(%)": [dtc_dep_dist_total],
                    prefix + "DTC Dependency Distance Mean(%)": [dtc_dep_dist_mean],
                    prefix + "DTC Discourse Total(%)": [dtc_disc_total], prefix + "DTC Discourse Rate(%)": [dtc_disc_r],
                    prefix + "DTC Information Units(%)": [dtc_iu_ner_total], prefix + "DTC Efficiency(%)": [dtc_iu_ner_eff],
                    prefix + "TF-IDF": [idf], prefix + "Accuracy12": [acc],
                    prefix + "Correct Count": [num_cor], }
        return pd.DataFrame(feat)


if __name__ == "__main__":
    sid = os.path.basename(DATA_PATH_EXAMPLE).split('_')[-1]
    audio_file_dt_pd = os.path.join(DATA_PATH_EXAMPLE, f"session_1/14_PDSS/{sid}_pd.wav")
    trans_file_dt_pd = os.path.join(TRANS_PATH_EXAMPLE, f"session_1/14_PDSS/{sid}_pd.txt")
    audio_file_dt_ss = os.path.join(DATA_PATH_EXAMPLE, f"session_1/14_PDSS/{sid}_ss.wav")
    trans_file_dt_ss = os.path.join(TRANS_PATH_EXAMPLE, f"session_1/14_PDSS/{sid}_ss.txt")
    audio_file_sig_pd = os.path.join(DATA_PATH_EXAMPLE, f"session_1/10_PD/CookieTheft/{sid}_CookieTheft.wav")
    trans_file_sig_pd = os.path.join(TRANS_PATH_EXAMPLE, f"session_1/10_PD/CookieTheft/{sid}_CookieTheft.txt")
    pdss_feat = PDSSFeatures(audio_file_dt_pd, trans_file_dt_pd, audio_file_dt_ss, trans_file_dt_ss,
                             audio_file_sig_pd, trans_file_sig_pd).get_all_feat()
    print(pdss_feat)
