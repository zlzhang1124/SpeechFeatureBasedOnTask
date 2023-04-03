#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/2/28 16:20
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : PD.py
# @Software : Python3.6; PyCharm; Windows10
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M
# @Version  : V1.4 - ZL.Z：2022/5/19
# 		      跟随SI.py修改
#             V1.3 - ZL.Z：2022/4/7
# 		      继承SI任务，添加相应特征
#             V1.2 - ZL.Z：2022/3/13
# 		      根据分析结果，将一些分类贡献度低、ICC低的特征做标记隐藏处理（方法前加_标记）
#             V1.0 - ZL.Z：2022/2/28
# 		      First version.
# @License  : None
# @Brief    : 提取PD图片描述任务的特征: 继承自我介绍SI任务的特征
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

from src.task.SI import SIFeatures
from src.config import *


class PDFeatures(SIFeatures):
    """获取PD任务的特征"""
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
        super(PDFeatures, self).__init__(input_f_audio, input_f_trans, f0min, f0max, sil_thr, min_sil, min_snd)

    def _duration_pause_intervals(self):
        """
        计算停顿间隔时间的中位值
        :return: dpi, float, ms
        """
        return self.duration_pause_intervals()

    def lexical_density(self):
        """
        计算该任务被试话语的词汇密度：词汇词(即实义词，包括实义动词v/名词n/形容词a)与总词汇数（不包括标点）比率
        :return: 词汇密度ld, float
        """
        return self._lexical_density()

    def _mean_len_utter(self):
        """
        计算平均话语长度（ Mean Length of Utterance，MLU）：每句话中词语的数量，即词数（不包括标点）与句子数之比
        :return: 词数与句子数之比，float
        """
        return self.mean_len_utter()

    def get_all_feat(self, prefix='PD_', use_=False):
        """
        获取当前所有特征
        :param prefix: pd.DataFrame类型特征列名的前缀
        :param use_: 是否提取包含分类贡献度低、ICC低的（方法有_标记）特征，默认否
        :return: 该类的全部特征, pd.DataFrame类型
        """
        f0_sd = self.f0_std()
        rst = self.rate_speech_timing()
        est = self.entropy_speech_timing()
        voiced_rate = self.voiced_rate()
        hesi_r = self.hesitation_ratio()
        mfcc = self.mfcc()
        wn = self.word_num_all()
        wr = self.word_rate()
        fwr = self.function_word_ratio()
        ld = self.lexical_density()
        sn = self.sentence_num()
        n_pr = self.noun_phrase_rate()
        v_pr = self.verb_phrase_rate()
        adj_pr = self.adj_phrase_rate()
        adv_pr = self.adv_phrase_rate()
        prep_pr = self.prep_phrase_rate()
        tree_h = self.parse_tree_height()
        yngve_max = self.max_yngve_depth()
        yngve_mean = self.mean_yngve_depth()
        yngve_total = self.total_yngve_depth()
        frazier_max = self.max_frazier_depth()
        frazier_mean = self.mean_frazier_depth()
        frazier_total = self.total_frazier_depth()
        dep_dist_total = self.total_dependency_distance()
        dep_dist_mean = self.mean_dependency_distance()
        disc_total = self.total_discourse_markers()
        disc_r = self.discourse_marker_rate()
        iu_ner_total = self.total_information_units_ner()
        iu_ner_eff = self.efficiency_information_units_ner()
        idf = self.tf_idf()
        if use_:
            int_sd = self._intensity_std()
            dpi = self._duration_pause_intervals()
            energy = self._energy_mean()
            mattr = self._lexical_diversity()
            mlu = self._mean_len_utter()
            feat = {prefix+"F0 SD(st)": [f0_sd], prefix+"Intensity SD(dB)": [int_sd], prefix+"DPI(ms)": [dpi],
                    prefix+"RST(-/s)": [rst], prefix+"EST": [est], prefix+"Voiced Rate(1/s)": [voiced_rate],
                    prefix+"Hesitation Ratio": [hesi_r], prefix+"Energy Mean(Pa^2·s)": [energy],
                    prefix+"MFCC": [mfcc], prefix+"Word Num": [wn],
                    prefix+"Word Rate(-/s)": [wr], prefix+"Function Word Ratio": [fwr], prefix+"Lexical Density": [ld],
                    prefix+"MATTR": [mattr], prefix+"Sentence Num": [sn], prefix+"MLU": [mlu],
                    prefix + "Noun Phrase Rate": [n_pr], prefix + "Verb Phrase Rate": [v_pr],
                    prefix + "Adj Phrase Rate": [adj_pr], prefix + "Adv Phrase Rate": [adv_pr],
                    prefix + "Prep Phrase Rate": [prep_pr], prefix + "Parse Tree Height": [tree_h],
                    prefix + "Yngve Depth Max": [yngve_max], prefix + "Yngve Depth Mean": [yngve_mean],
                    prefix + "Yngve Depth Total": [yngve_total], prefix + "Frazier Depth Max": [frazier_max],
                    prefix + "Frazier Depth Mean": [frazier_mean], prefix + "Frazier Depth Total": [frazier_total],
                    prefix + "Dependency Distance Total": [dep_dist_total],
                    prefix + "Dependency Distance Mean": [dep_dist_mean],
                    prefix + "Discourse Total": [disc_total], prefix + "Discourse Rate": [disc_r],
                    prefix + "Information Units": [iu_ner_total], prefix + "Efficiency": [iu_ner_eff],
                    prefix+"TF-IDF": [idf], }
        else:
            feat = {prefix + "F0 SD(st)": [f0_sd], prefix + "RST(-/s)": [rst], prefix + "EST": [est],
                    prefix + "Voiced Rate(1/s)": [voiced_rate], prefix+"Hesitation Ratio": [hesi_r],
                    prefix + "MFCC": [mfcc], prefix + "Word Num": [wn],
                    prefix + "Word Rate(-/s)": [wr], prefix + "Function Word Ratio": [fwr],
                    prefix + "Lexical Density": [ld], prefix + "Sentence Num": [sn],
                    prefix + "Noun Phrase Rate": [n_pr], prefix + "Verb Phrase Rate": [v_pr],
                    prefix + "Adj Phrase Rate": [adj_pr], prefix + "Adv Phrase Rate": [adv_pr],
                    prefix + "Prep Phrase Rate": [prep_pr], prefix + "Parse Tree Height": [tree_h],
                    prefix + "Yngve Depth Max": [yngve_max], prefix + "Yngve Depth Mean": [yngve_mean],
                    prefix + "Yngve Depth Total": [yngve_total], prefix + "Frazier Depth Max": [frazier_max],
                    prefix + "Frazier Depth Mean": [frazier_mean], prefix + "Frazier Depth Total": [frazier_total],
                    prefix + "Dependency Distance Total": [dep_dist_total],
                    prefix + "Dependency Distance Mean": [dep_dist_mean],
                    prefix + "Discourse Total": [disc_total], prefix + "Discourse Rate": [disc_r],
                    prefix + "Information Units": [iu_ner_total], prefix + "Efficiency": [iu_ner_eff],
                    prefix + "TF-IDF": [idf], }
        return pd.DataFrame(feat)


if __name__ == "__main__":
    sid = os.path.basename(DATA_PATH_EXAMPLE).split('_')[-1]
    audio_file = os.path.join(DATA_PATH_EXAMPLE, f"session_1/10_PD/CookieTheft/{sid}_CookieTheft.wav")
    trans_file = os.path.join(TRANS_PATH_EXAMPLE, f"session_1/10_PD/CookieTheft/{sid}_CookieTheft.txt")
    pd_feat = PDFeatures(audio_file, trans_file).get_all_feat()
    print(pd_feat)
