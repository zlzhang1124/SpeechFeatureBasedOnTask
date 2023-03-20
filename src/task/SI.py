#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2021/5/21 17:27
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: 1. V. Boschi, E. Catricala, M. Consonni, C. Chesi, A. Moro, and S. F. Cappa,
#             "Connected Speech in Neurodegenerative Language Disorders: A Review," Front Psychol, vol. 8, p. 269, 2017.
#             2. A. Shivkumar, J. Weston, R. Lenain, and E. Fristed, "BlaBla: Linguistic Feature Extraction for
#             Clinical Analysis in Multiple Languages," presented at the Interspeech 2020, 2020.
# @FileName : 14_SI.py
# @Software : Python3.6; PyCharm; Windows10
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M
# @Version  : V1.4 - ZL.Z：2022/5/19
# 		      1. 使用To TextGrid (silences)方法替换To TextGrid (vuv)，仅分割有声和无声段;
# 		      2. 总用时改为录制的实际音频时长，不去头去尾
#             V1.3 - ZL.Z：2022/3/31 - 2022/4/7
# 		      1. 添加基于声学的特征：Hesitation Ratio;
# 		      2. 添加句法特征（Morpho-syntactic & Syntactic）：Noun Phrase Rate/Verb Phrase Rate/Adj Phrase Rate/
# 		      Adv Phrase Rate/Prep Phrase Rate/Parse Tree Height/Yngve Depth Max/Yngve Depth Mean/
# 		      Yngve Depth Total/Frazier Depth Max/Frazier Depth Mean/Frazier Depth Total/
# 		      Dependency Distance Total/Dependency Distance Mean;
# 		      3. 添加语篇和语用（Discourse & Pragmatic）特征：Discourse Total/Discourse Rate/Information Units/Efficiency.
#             V1.2 - ZL.Z：2022/3/13
# 		      根据分析结果，将一些分类贡献度低、ICC低(>0.6)的特征做标记隐藏处理（方法前加_标记）
#             V1.1 - ZL.Z：2022/2/27, 2022/3/3
# 		      1. vup_duration_from_vuvTextGrid改为vup_duration_from_vuvInfo
#             2. 添加部分文本特征：
#             V1.0 - ZL.Z：2021/5/21
# 		      First version.
# @License  : None
# @Brief    : 提取SI自我介绍任务的特征:
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

from src.utils.util import *
from src.config import *
import re
import numpy as np
import librosa
import jieba
import jieba.posseg
import hanlp
from sklearn.feature_extraction.text import TfidfVectorizer

jieba.setLogLevel(jieba.logging.INFO)
# 预训练模型,采用百度的ERNIE-Gram中文MTL模型(SuperGLUE曾经的榜首)，
# HanLP的Native API输入单位为句子，需使用多语种分句模型或基于规则的分句函数先行分句
# devices=-1仅用CPU，以避免多线程调用GPU报错（RuntimeError: Cannot re-initialize CUDA in forked subprocess.
# To use CUDA with multiprocessing, you must use the 'spawn' start method）
DEVICE = None  # 若使用全部GPU则DEVICE = None，此时不能并行提取该任务特征；设置devices=-1仅用CPU，则可以并行（这里python原生Hanlp无法并行，程序会卡住不动）
HanLP = hanlp.pipeline().append(hanlp.utils.rules.split_sentence, output_key='sentences')\
    .append(hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ERNIE_GRAM_ZH, devices=DEVICE), output_key='ernie')
# 预训练模型,采用Electra (Clark. 2020)基于细粒度CWS语料训练的小模型，得到tok
# 输入在CTB9-UD420上训练的Electra小型编码器(Clark，2020)与Biaffine译码器(Dozat & Manning 2017)。
#  (以获取基于Universal Dependencies中依存句法语篇discourse标记)，
HanLP_UD = hanlp.pipeline().append(hanlp.utils.rules.split_sentence, output_key='sentences')\
    .append(hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH, devices=DEVICE), output_key='tok') \
    .append(hanlp.load(hanlp.pretrained.dep.CTB9_UDC_ELECTRA_SMALL, devices=DEVICE), output_key='dep', input_key='tok')


class SIFeatures:
    """获取SI任务的特征"""
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
        self.f_audio = input_f_audio
        self.f_trans = input_f_trans
        self.f0min = f0min
        self.f0max = f0max
        self.sound = parselmouth.Sound(self.f_audio)
        self.total_duration = self.sound.get_total_duration()
        self.text_grid_vuv = call(self.sound, "To TextGrid (silences)", 100, 0.0, sil_thr, min_sil, min_snd, 'U', 'V')
        self.vuv_info = call(self.text_grid_vuv, "List", False, 10, False, False)
        self.text = ''
        if self.f_trans.endswith('.txt'):  # 自动转录
            try:
                with open(self.f_trans, "r", encoding="utf-8") as f:
                    self.text = f.read()
            except UnicodeDecodeError:
                with open(self.f_trans, "r", encoding="gb18030") as f:
                    self.text = f.read()
            # 自动转录根据所定义规则分割句子，句子间用中文句号分割，以用于传入hanlp的pipeline中句子分割层
            self.sent_text = u'。'.join(self.func_split_sentence())
        elif self.f_trans.endswith('.cha'):
            self.sent_text = self.text  # 手工转录，默认句子间的标点分割完全正确
        self.text_no_punct = delete_punctuation(self.text)  # 删除标点符号后的文本
        self.text_seg_list = jieba.lcut(self.text)  # 分词结果列表
        self.text_seg_list_no_punct = delete_punctuation(self.text_seg_list)  # 删除标点符号后的分词结果
        self.text_posseg_list = jieba.posseg.lcut(self.text)  # 分词结果列表，含词性(包含所有词性)
        self.sent_num = len(HanLP(self.sent_text)['sentences'])
        self.doc = HanLP(self.sent_text)['ernie']  # 基于多任务学习模型的全部结果，Document类型，该结果用于获取除了语篇的其他特征
        # 基于CTB9-UD420语料库的依存句法结果，list类型，元素为每句话，CoNLLSentence类型，该结果仅用于获取语篇相关特征
        self.dep_ud = HanLP_UD(self.sent_text)['dep']

    def func_split_sentence(self, sep=None):
        """
        对给定文本，根据标点分割成句子：默认逗号、句号、问号、感叹号、分号、省略号分割的都为一个句子
        :param sep: 设定的句子分割标点，默认[',', '\.', '\?', '\!', ';', '，', '。', '？', '！', '；']，特殊字符需转义\
        :return: 句子列表
        """
        if sep is None:
            sep = [',', '\.', '\?', '\!', ';', '，', '。', '？', '！', '；']
        pattern = re.compile('|'.join(sep))
        text = re.sub(pattern, '*', self.text)
        sent_list = list(filter(None, text.split('*')))  # 去除空元素之后的句子列表
        return sent_list

    def func_phrase_rate(self, tree_tag: str):
        """
        计算特定类型短语比例：特定类型短语数量/句子数量，其中特定类型短语设定为至少包含一个特定类型及其附属词的特定类型短语，
        且为最大长度，即不包含特定类型短语中的特定类型短语
        :param tree_tag: 基于Chinese Tree Bank的短语类型标签，参见https://hanlp.hankcs.com/docs/annotations/constituency/ctb.html
        :return: npr，特定类型短语率
        """
        con = self.doc['con']  # 该语篇包含多条句子，con为phrasetree.tree.Tree短语结构树类型
        np_l = []
        for child in con:  # 对于每一个子树
            last_str = ''
            for subtree in child.subtrees(lambda t: t.label() == tree_tag):  # 查找标签为tree_tag的特定类型短语
                # 至少包含一个特定类型及其附属词的特定类型短语，且不包含特定类型短语中的特定类型短语
                if len(subtree.leaves()) > 1 and ''.join(subtree.leaves()) not in last_str:
                    np_l.append(subtree.leaves())
                last_str = ''.join(subtree.leaves())
        return len(np_l) / self.sentence_num()

    def func_calc_yngve_score(self, tree, parent):
        """
        递归计算句法复杂度指标：Yngve评分
        ref: B. Roark, M. Mitchell, and K. Hollingshead, "Syntactic complexity measures for detecting Mild Cognitive
        Impairment," presented at the Proceedings of the Workshop on BioNLP 2007, 2007.
        https://github.com/meyersbs/SPLAT/blob/fd211e49582c64617d509db5746b99075a25ad9b/splat/complexity/__init__.py#L279
        https://github.com/neubig/util-scripts/blob/96c91e43b650136bb88bbb087edb1d31b65d389f/syntactic-complexity.py
        :param tree: phrasetree.tree.Tree类型短语结构树
        :param parent: 父节点评分，初始调用为0
        :return: Yngve评分
        """
        if type(tree) == str:
            return parent
        else:
            count = 0
            for i, child in enumerate(reversed(tree)):
                count += self.func_calc_yngve_score(child, parent + i)
            return count

    def func_get_yngve_list(self):
        """
        获取整个文本所有子树的yngve评分列表，其中每个子树的yngve=全部叶子的yngve和/叶子数，即在叶子尺度上的平均yngve得分
        :return: 整个文本所有子树的yngve评分列表
        """
        con = self.doc['con']  # 该语篇包含多条句子，con为phrasetree.tree.Tree短语结构树类型
        yngve_l = []
        for child in con:  # 对于每一个子树
            if child.label() != 'PU':  # 排除仅包含一个标点符号的子树
                yngve_l.append(self.func_calc_yngve_score(child, 0) / len(child.leaves()))  # 计算该结构树的yngve得分
        return yngve_l

    def func_calc_frazier_score(self, tree, parent, parent_label):
        """
        递归计算句法复杂度指标：Frazier评分
        ref: B. Roark, M. Mitchell, and K. Hollingshead, "Syntactic complexity measures for detecting Mild Cognitive
        Impairment," presented at the Proceedings of the Workshop on BioNLP 2007, 2007.
        https://github.com/meyersbs/SPLAT/blob/fd211e49582c64617d509db5746b99075a25ad9b/splat/complexity/__init__.py#L279
        https://github.com/neubig/util-scripts/blob/96c91e43b650136bb88bbb087edb1d31b65d389f/syntactic-complexity.py
        :param tree: phrasetree.tree.Tree类型短语结构树
        :param parent: 父节点评分，初始调用为0
        :param parent_label: 父节点标记，初始调用为“”
        :return: Frazier评分
        """
        my_lab = ''
        if type(tree) == str:
            return parent - 1
        else:
            count = 0
            for i, child in enumerate(tree):
                score = 0
                if i == 0:
                    my_lab = tree.label()
                    if my_lab == "IP" or my_lab.startswith('S'):  # 句子节点
                        if parent_label == "IP" or my_lab.startswith('S'):
                            score = 0
                        else:
                            score = parent + 1.5
                    elif my_lab != "" and my_lab != "ROOT" and my_lab != "TOP":
                        score = parent + 1
                count += self.func_calc_frazier_score(child, score, my_lab)
            return count

    def func_get_frazier_list(self):
        """
        获取整个文本所有子树的frazier评分列表，其中每个子树的frazier=全部叶子的frazier和/叶子数，即在叶子尺度上的平均frazier得分
        :return: 整个文本所有子树的frazier评分列表
        """
        con = self.doc['con']  # 该语篇包含多条句子，con为phrasetree.tree.Tree短语结构树类型
        frazier_l = []
        for child in con:  # 对于每一个子树
            if child.label() != 'PU':  # 排除标点符号
                frazier_l.append(self.func_calc_frazier_score(child, 0, '') / len(child.leaves()))  # 计算该结构树的frazier得分
        return frazier_l

    def f0_std(self):
        """
        计算基频的标准偏差
        :return: f0_sd, float, semitones
        """
        pitch_obj = call(self.sound, "To Pitch", 0.0, self.f0min, self.f0max)
        f0_sd = call(pitch_obj, "Get standard deviation", 0.0, 0.0, "semitones")
        return f0_sd

    def _intensity_std(self):
        """
        计算浊音段声音强度的标准差
        :return: int_sd, float, dB
        """
        segments_v, segments_u = duration_from_vuvInfo(self.vuv_info)
        intensity_obj = call(self.sound, "To Intensity", 100, 0.0)
        int_list = []
        for seg in segments_v:  # 遍历浊音段
            frame_start = round(call(intensity_obj, "Get frame number from time", seg[0]))  # 时间点转帧数
            frame_end = round(call(intensity_obj, "Get frame number from time", seg[1]))
            for frame in range(frame_start, frame_end+1):
                int_val = call(intensity_obj, "Get value in frame", frame)
                if pd.notna(int_val):
                    int_list.append(int_val)
        int_sd = float(np.std(int_list))
        return int_sd

    def duration_pause_intervals(self):
        """
        计算停顿间隔时间的中位值
        :return: dpi, float, ms
        """
        segments_v, segments_u = duration_from_vuvInfo(self.vuv_info)
        duration_p = np.array(segments_u)[:, 1] - np.array(segments_u)[:, 0]
        dpi = float(1000 * np.median(duration_p))
        return dpi

    def rate_speech_timing(self):
        """
        计算计时独白演讲的速率：有声和停顿间隔数与时间进程的比率，以每次总间隔计数与时间的回归线的斜率进行测量。
        （回归：纵坐标为有声和停顿间隔段数量的累加和，横坐标为时间，两者回归取回归线斜率即为RST）
        :return: rst, float, -/s
        """
        segments_v, segments_u = duration_from_vuvInfo(self.vuv_info)
        total_seg = None
        fir_flag = True
        for i_seg in segments_v, segments_u:  # 合并所有段,并避免有的段为空
            if fir_flag and i_seg:
                total_seg = np.array(i_seg)
                fir_flag = False
                continue
            if i_seg:
                total_seg = np.vstack((np.array(total_seg), np.array(i_seg)))
        total_seg = total_seg.flatten()  # 展平成一维
        total_seg.sort()  # 从小到大排序
        total_seg = np.unique(total_seg)  # 删除重复数据
        y_num = [i for i in range(len(total_seg))]  # 每增一个时间点此时就会增加一个间隔段
        x_time = total_seg  # 时间轴
        rst = np.polyfit(x=x_time, y=y_num, deg=1)[0]  # 线性回归，取一次项系数作为RST
        return rst

    def entropy_speech_timing(self):
        """
        计算计时独白演讲的香农信息熵：从有声段和停顿段上计算
        :return: est, float
        """
        segments_v, segments_u = duration_from_vuvInfo(self.vuv_info)
        total_seg_num = len(segments_v) + len(segments_u)
        prob1 = len(segments_v) / total_seg_num
        prob2 = len(segments_u) / total_seg_num
        est = - prob1 * np.log2(prob1, out=np.zeros(1), where=(prob1 > 0))[0] - \
              prob2 * np.log2(prob2, out=np.zeros(1), where=(prob2 > 0))[0]  # 使用out和where，避免log出错
        return est

    def voiced_rate(self):
        """
        计算SI任务的语音速率：每秒出现的浊音段数量
        :return: rate, float, 1/s
        """
        segments_v, segments_u = duration_from_vuvInfo(self.vuv_info)
        rate = len(segments_v) / self.total_duration
        return rate

    def hesitation_ratio(self):
        """
        计算SI任务的犹豫率：犹豫的总持续时间除以总演讲时间，其中犹豫被定义为持续超过30毫秒的没有说话
        ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
        :return: hesi_ratio, float
        """
        segments_v, segments_u = duration_from_vuvInfo(self.vuv_info)
        try:
            duration_p = np.array(segments_u)[:, 1] - np.array(segments_u)[:, 0]
            hesi_ratio = np.sum(duration_p[duration_p > 0.03]) / self.total_duration
            return hesi_ratio
        except IndexError:
            if not len(segments_u):
                return 0.0
            elif not len(segments_v):
                return 1.0

    def _energy_mean(self):
        """
        计算浊音段语音的平均能量
        :return: energy, float, Pa^2·s
        """
        segments_v, segments_u = duration_from_vuvInfo(self.vuv_info)
        e_list = []
        for seg in segments_v:  # 遍历浊音段
            e_list.append(call(self.sound, "Get energy", seg[0], seg[1]))
        energy = float(np.mean(e_list))
        return energy

    def mfcc(self):
        """
        计算39维MFCC系数：13个MFCC特征（第一个系数为能量c0）及其对应的一阶和二阶差分
        :return: 13*3维MFCC特征，每一列为一个MFCC特征向量 np.ndarray[shape=(n_mfcc*3, n_frames), dtype=float64]
        """
        mfcc_obj = self.sound.to_mfcc(number_of_coefficients=12, window_length=0.015, time_step=0.005,
                                      firstFilterFreqency=100.0, distance_between_filters=100.0)
        mfcc_f = mfcc_obj.to_array()
        mfcc_delta1 = librosa.feature.delta(mfcc_f)  # 一阶差分
        mfcc_delta2 = librosa.feature.delta(mfcc_f, order=2)  # 二阶差分
        mfcc_f = np.vstack((mfcc_f, mfcc_delta1, mfcc_delta2))  # 整合成39维MFCC特征
        return mfcc_f

    def word_num_all(self):
        """
        计算该任务被试话语中包含的所有词语数量（不包括标点）
        :return: 所有词语数量, int
        """
        return len(self.text_seg_list_no_punct)

    def word_rate(self):
        """
        计算该任务被试话语的速率：每秒词语数
        :return: 每秒词语数，单位词/s
        """
        return self.word_num_all() / self.total_duration

    def function_word_ratio(self):
        """
        计算该任务被试话语中虚词(包括副词d/介词p/连词c/助词u/叹词e/语气词y/拟声词o，不包括非语素词x和标点w)与所有词（不包括标点）的比值
        :return: 虚词与所有词（不包括标点）的比值fw_ratio, float
        """
        func_w_l = ['p', 'u', 'ud', 'uj', 'uv', 'ug', 'ul', 'uz', 'y', 'd', 'dg', 'h', 't', 'zg', 'c', 'e', 'o']  # jieba的虚词标注
        func_w = [(x.word, x.flag) for x in self.text_posseg_list if x.flag in func_w_l]
        fw_ratio = len(func_w) / len(self.text_seg_list_no_punct)
        return fw_ratio

    def _lexical_density(self):
        """
        计算该任务被试话语的词汇密度：词汇词(即实义词，包括实义动词v/名词n/形容词a)与总词汇数（不包括标点）比率
        :return: 词汇密度ld, float
        """
        noun_l = ['n', 'j', 'mg', 'ng', 'nr', 'nrfg', 'nrt', 'ns', 'nt', 'nz', 's', 'tg']  # jieba的名词标注
        verb_l = ['v', 'df', 'f', 'vg', 'vd', 'vi', 'vn', 'vq']  # jieba的动词标注
        adj_l = ['a', 'ag', 'ad', 'an', 'b', 'i', 'z']  # jieba的形容词标注
        lw_l = noun_l + verb_l + adj_l
        lw = [(x.word, x.flag) for x in self.text_posseg_list if x.flag in lw_l]
        ld = len(lw) / len(self.text_seg_list_no_punct)
        return ld

    def _lexical_diversity(self, win_len=50):
        """
        计算该任务被试话语的词汇多样性（以移动平均形符比，Moving Average Type-Token Ratio, MATTR表示）(Covington and Mcfall, 2010)
        TTR:类符（非重复词）与形符（所有词，不包括标点）之比
        MATTR: 使用固定长度的窗口（如10个单词）,滑动窗口中类符的数量除以形符。
        例如，估计第1-10个单词的TTR，然后是单词2-11的TTR，然后是单词3-12，以此类推。最终对所有的TTR进行平均。
        MATTR计算方法能够减少文本长度对标准TTR数值的影响
        :param win_len: 滑动窗口长度，单位为词数，默认50个词
        :return: MATTR, float
        """
        assert (type(win_len) is int) and (win_len > 0), "win_len必须为正整数"
        if win_len >= len(self.text_seg_list_no_punct):  # 文本长度短于窗长，返回TTR
            mattr = len(sorted(set(self.text_seg_list_no_punct), key=self.text_seg_list_no_punct.index)) / \
                    len(self.text_seg_list_no_punct)
            return mattr
        ttr_win = []
        for i_win in range(len(self.text_seg_list_no_punct)):
            i_token = self.text_seg_list_no_punct[i_win:win_len]
            i_type = sorted(set(i_token), key=i_token.index)  # 去重并保持原序
            ttr_win.append(len(i_type) / len(i_token))
            win_len += 1
            if win_len >= len(self.text_seg_list_no_punct):
                break
        mattr = sum(ttr_win) / len(ttr_win)
        return mattr

    def sentence_num(self):
        """
        获取句子数：默认逗号、句号、问号、感叹号、分号、省略号分割的都为一个句子
        :return: 句子数量, int
        """
        return self.sent_num

    def mean_len_utter(self):
        """
        计算平均话语长度（ Mean Length of Utterance，MLU）：每句话中词语的数量，即词数（不包括标点）与句子数之比
        :return: 词数与句子数之比，float
        """
        return self.word_num_all() / self.sentence_num()

    def noun_phrase_rate(self):
        """
        计算名词短语比例：名词短语数量/句子数量，其中名词短语设定为至少包含一个名词及其附属词的名词短语，且为最大长度，即不包含名词短语中的名词短语
        :return: npr，名词短语率
        """
        return self.func_phrase_rate('NP')

    def verb_phrase_rate(self):
        """
        计算动词短语比例：动词短语数量/句子数量，其中动词短语设定为至少包含一个动词及其附属词的动词短语，且为最大长度，即不包含动词短语中的动词短语
        :return: npr，动词短语率
        """
        return self.func_phrase_rate('VP')

    def adj_phrase_rate(self):
        """
        计算形容词短语比例：形容词短语数量/句子数量，其中形容词短语设定为至少包含一个形容词及其附属词的形容词短语，
        且为最大长度，即不包含形容词短语中的形容词短语
        :return: npr，形容词短语率
        """
        return self.func_phrase_rate('ADJP')

    def adv_phrase_rate(self):
        """
        计算副词短语比例：副词短语数量/句子数量，其中副词短语设定为至少包含一个副词及其附属词的副词短语，
        且为最大长度，即不包含副词短语中的副词短语
        :return: npr，副词短语率
        """
        return self.func_phrase_rate('ADVP')

    def prep_phrase_rate(self):
        """
        计算介词短语比例：介词短语数量/句子数量，其中介词短语设定为至少包含一个介词及其附属词的介词短语，且为最大长度，即不包含介词短语中的介词短语
        :return: npr，介词短语率
        """
        return self.func_phrase_rate('PP')

    def parse_tree_height(self):
        """
        计算结构树的平均高度：在全部句子中，结构子树高度的平均值（不包括标点符号）
        :return: 结构树的平均高度
        """
        con = self.doc['con']  # 该语篇包含多条句子，con为phrasetree.tree.Tree短语结构树类型
        pth_l = []
        for child in con:  # 对于每一个子树
            if child.label() != 'PU':  # 排除标点符号
                pth_l.append(child.height())
        return np.mean(pth_l)

    def max_yngve_depth(self):
        """
        计算所有子树的最大Yngve深度：在全部句子中，结构子树Yngve深度的最大值
        :return: 所有子树的最大Yngve深度
        """
        return max(self.func_get_yngve_list())

    def mean_yngve_depth(self):
        """
        计算所有子树的平均Yngve深度：在全部句子中，结构子树Yngve深度的平均值
        :return: 所有子树的平均Yngve深度
        """
        return np.mean(self.func_get_yngve_list())

    def total_yngve_depth(self):
        """
        计算所有子树的Yngve深度总和：在全部句子中，结构子树Yngve深度的总和
        :return: 所有子树的Yngve深度总和
        """
        return sum(self.func_get_yngve_list())

    def max_frazier_depth(self):
        """
        计算所有子树的最大Frazier深度：在全部句子中，结构子树Frazier深度的最大值
        :return: 所有子树的最大Frazier深度
        """
        return max(self.func_get_frazier_list())

    def mean_frazier_depth(self):
        """
        计算所有子树的平均Frazier深度：在全部句子中，结构子树Frazier深度的平均值
        :return: 所有子树的平均Frazier深度
        """
        return np.mean(self.func_get_frazier_list())

    def total_frazier_depth(self):
        """
        计算所有子树的Frazier深度总和：在全部句子中，结构子树Frazier深度的总和
        :return: 所有子树的Frazier深度总和
        """
        return sum(self.func_get_frazier_list())

    def total_dependency_distance(self):
        """
        根据依存句法结果，计算在全部句子上平均总的依存距离：每个句子的总依存距离之和/句子数，
        其中依存距离定义为每条依存连接的支配词（中心词）与从属词（修饰词）之间的距离
        ref: B. Roark, M. Mitchell, and K. Hollingshead, "Syntactic complexity measures for detecting Mild
        Cognitive Impairment," presented at the Proceedings of the Workshop on BioNLP 2007, 2007.
        :return: 全部句子上总依存距离的平均值
        """
        dep = self.doc['dep']  # 所有句子的依存句法树列表，第i个二元组表示第i个单词的[中心词的下标, 与中心词的依存关系]
        sen_dep = []
        for i_dep in dep:
            i_dep_total = 0
            for j_dep in i_dep:
                if j_dep[0] != 0:
                    i_dep_total += abs(i_dep.index(j_dep) + 1 - j_dep[0])
            sen_dep.append(i_dep_total)
        return np.mean(sen_dep)

    def mean_dependency_distance(self):
        """
        根据依存句法结果，计算在全部句子上平均依存距离的平均值：每个句子的平均依存距离之和/句子数，其中平均依存距离=每个句子的总依存距离/依存连接数
        其中依存距离定义为每条依存连接的支配词（中心词）与从属词（修饰词）之间的距离
        ref: B. Roark, M. Mitchell, and K. Hollingshead, "Syntactic complexity measures for detecting Mild
        Cognitive Impairment," presented at the Proceedings of the Workshop on BioNLP 2007, 2007.
        :return: 全部句子上平均依存距离的平均值
        """
        dep = self.doc['dep']  # 所有句子的依存句法树列表，第i个二元组表示第i个单词的[中心词的下标, 与中心词的依存关系]
        sen_dep_mean = []
        for i_dep in dep:
            i_dep_total = 0
            dep_link = 0
            for j_dep in i_dep:
                if j_dep[0] != 0:
                    dep_link += 1
                    i_dep_total += abs(i_dep.index(j_dep) + 1 - j_dep[0])
            try:
                sen_dep_mean.append(i_dep_total / dep_link)
            except ZeroDivisionError:
                pass
        return np.mean(sen_dep_mean)

    def total_discourse_markers(self):
        """
        根据依存句法分析结果，获取全部文本中语篇标记discourse的总数量，UD语料库中存在该标记，因此这里使用基于CTB9-UD420语料库的依存句法结果
        :return: 全部文本中语篇标记discourse的总数量
        """
        num_disc = 0
        for i_sen in self.dep_ud:  # 针对每一句话
            for j_wd in i_sen:  # 每一个词
                if j_wd.deprel.startswith('discourse'):  # 语篇标记
                    num_disc += 1
        return num_disc

    def discourse_marker_rate(self):
        """
        根据依存句法分析结果，计算平均每句话的语篇/话语标记discourse的数量，UD语料库中存在该标记，因此这里使用基于CTB9-UD420语料库的依存句法结果
        :return: 每句话的语篇标记discourse的数量
        """
        return self.total_discourse_markers() / self.sentence_num()

    def total_information_units_ner(self):
        """
        基于命名实体识别NER分析结果，获取全部文本中的命名实体数量，以此作为信息单元（这里采用OntoNotes标注标准）
        :return: 全部的信息单元数量（命名实体数量）
        """
        num_iu = 0
        for i_sen in self.doc['ner/ontonotes']:  # 针对每一句话
            num_iu += len(i_sen)
        return num_iu

    def efficiency_information_units_ner(self):
        """
        基于命名实体识别NER分析结果，计算信息单元的效率：信息单元总数/语音样本的持续时间（这里采用OntoNotes标注标准）
        :return: 信息单元的效率
        """
        return self.total_information_units_ner() / self.total_duration

    def tf_idf(self, stop_words=None):
        """
        计算文本的词频-逆文档频率TF-IDF
        :param stop_words: 过滤的停用词列表，默认['是', '的', '了']
        :return: tf_idf, np.narray[(n_words(排除标点、重复、停用词),), float64]
        """
        if stop_words is None:
            stop_words = ['是', '的', '了']
        text = ' '.join(i for i in self.text_seg_list)
        tfidf = TfidfVectorizer(stop_words=stop_words, token_pattern=r"(?u)\b\w+\b")  # 排除标点和停用词，包括单字
        tf_idf = tfidf.fit_transform([text]).toarray()[0]
        return tf_idf

    def get_all_feat(self, prefix='SI_', use_=False):
        """
        获取当前所有特征
        :param prefix: pd.DataFrame类型特征列名的前缀
        :param use_: 是否提取包含分类贡献度低、ICC低的（方法有_标记）特征，默认否
        :return: 该类的全部特征, pd.DataFrame类型
        """
        f0_sd = self.f0_std()
        dpi = self.duration_pause_intervals()
        rst = self.rate_speech_timing()
        est = self.entropy_speech_timing()
        voiced_rate = self.voiced_rate()
        hesi_r = self.hesitation_ratio()
        mfcc = self.mfcc()
        wn = self.word_num_all()
        wr = self.word_rate()
        fwr = self.function_word_ratio()
        sn = self.sentence_num()
        mlu = self.mean_len_utter()
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
            energy = self._energy_mean()
            ld = self._lexical_density()
            mattr = self._lexical_diversity()
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
            feat = {prefix+"F0 SD(st)": [f0_sd], prefix+"DPI(ms)": [dpi],
                    prefix+"RST(-/s)": [rst], prefix+"EST": [est], prefix+"Voiced Rate(1/s)": [voiced_rate],
                    prefix+"Hesitation Ratio": [hesi_r], prefix+"MFCC": [mfcc], prefix+"Word Num": [wn],
                    prefix+"Word Rate(-/s)": [wr], prefix+"Function Word Ratio": [fwr],
                    prefix+"Sentence Num": [sn], prefix+"MLU": [mlu],
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
        return pd.DataFrame(feat)


if __name__ == "__main__":
    sid = os.path.basename(DATA_PATH_EXAMPLE).split('_')[-1]
    audio_file = os.path.join(DATA_PATH_EXAMPLE, f"session_1/12_SI/{sid}_si.wav")
    trans_file = os.path.join(TRANS_PATH_EXAMPLE, f"session_1/12_SI/{sid}_si.txt")
    si_feat = SIFeatures(audio_file, trans_file).get_all_feat()
    print(si_feat)
