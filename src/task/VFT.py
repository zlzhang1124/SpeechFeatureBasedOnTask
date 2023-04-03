#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/3/14 15:20 
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: 1.  N. Linz, K. L. Fors, H. Lindsay, M. Eckerström, J. Alexandersson, and D. Kokkinakis,
#             "Temporal Analysis of the Semantic Verbal Fluency Task in Persons with Subjective and Mild Cognitive
#             Impairment," in Proceedings of the Sixth Workshop on Computational Linguistics and Clinical Psychology.
#             Association for Computational Linguistics, 2019, pp. 103–113.
#             2.  Hali Lindsay, Philipp Müller, Nicklas Linz, Radia Zeghari, Mario Magued Mina, Alexandra Konig,
#             and J. Tröger, "Dissociating Semantic and Phonemic Search Strategies in the Phonemic Verbal Fluency Task
#             in early Dementia," in Proceedings ofthe Seventh Workshop on Computational Linguistics and Clinical
#             Psychology. Association for Computational Linguistics, 2021, pp. 32–44.
# @FileName : VFT.py
# @Software : Python3.6; PyCharm; Windows10 / Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M / 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090
# @Version  : V1.0 - ZL.Z：2022/3/21 - 2022/3/24
# 		      First version.
# @License  : None
# @Brief    : VFT言语流畅性相关特征（全部基于文本）：分别PFT/SFT的Word Count/Repetitions/Intrusions/WC Bin/TD Bin/
#                   SD Bin/LD Bin/Cluster Size/Cluster Switches/Word Frequency(%)

import regex
import numpy as np
import jieba
import jieba.posseg
import pypinyin
from pypinyin_dict.pinyin_data import cc_cedict
import fasttext
from scipy.spatial import distance
import epitran
import epitran.vector
import editdistance
from src.utils.util import *
from src.config import *
jieba.setLogLevel(jieba.logging.INFO)
cc_cedict.load()  # 使用来源 cc-cedict.org 的词语拼音数据生成的单个汉字拼音数据，避免原始多音字包含不常见甚至错误的音
fasttext.FastText.eprint = lambda x: None
zhong = regex.compile(r'^\p{Han}*$')  # 匹配中文


ANIMAL_DICT = os.path.join(parent_path, "dicts/animal.txt")
with open(ANIMAL_DICT, "r", encoding="utf-8") as _f:
    ANIMAL = {}
    for i_con in _f.readlines():
        if (i_con.strip() not in ANIMAL) and i_con.strip():
            ani_pinyin = ' '.join(pypinyin.lazy_pinyin(i_con.strip(), style=pypinyin.Style.TONE3))
            ANIMAL[i_con.strip()] = ani_pinyin  # 带有拼音的动物词汇字典，键为动物词汇汉字，值为对应的拼音
STOP_WORDS = os.path.join(parent_path, "dicts/stop_words_vft.txt")
STOPWORDS = [line.strip() for line in open(STOP_WORDS, encoding='utf-8').readlines()]
ChineseWordFrequency = os.path.join(parent_path, "dicts/CorpusWordlist.csv")  # 现代汉语语料库词频表
with open(ChineseWordFrequency, "r", encoding="utf-8") as _f:
    CWF = {}
    for i_con in _f.readlines():
        CWF[i_con.strip().split(',')[1]] = float(i_con.strip().split(',')[-1])
CEDICT = os.path.join(parent_path, "dicts/cedict_ts.u8")  # 汉英词典
ALIGN_DICT = os.path.join(P2FA_MODEL, "dict")  # P2FA的字典，对齐用
with open(ALIGN_DICT, "r", encoding="utf-8") as _f:
    ALIGN = {}
    for i_con in _f.readlines():
        if (i_con.strip() not in ALIGN) and i_con.strip() and len(i_con.split('  ')[0]) == 1:
            ali_pinyin = ' '.join(pypinyin.lazy_pinyin(i_con.split('  ')[0], style=pypinyin.Style.TONE3))
            # 带有拼音的对齐词汇字典，键为对应的拼音，值为对应的音素
            ALIGN[ali_pinyin] = i_con.strip().replace(i_con.split('  ')[0], '')
MODEL_fastText = fasttext.load_model(PRETRAINED_MODEL_fastText)  # 基于维基百科的300d中文词向量预训练模型


class VFTFeatures:
    """获取VFT任务的特征"""
    def __init__(self, input_f_audio: str, input_f_trans: str, temp_dir: str, miss_w_dir: str, is_pf=None):
        """
        初始化
        :param input_f_audio: 输入.wav音频文件，或是praat所支持的文件格式
        :param input_f_trans: 输入文本转录文件，.txt或.cha类似的文件格式
        :param temp_dir: 音频/文本对齐的中间结果临时保存路径
        :param miss_w_dir: 所用的字典中未存在的丢失词文件保存路径（包括对齐用的字典和动物词汇字典）
        :param is_pf: 是否为PFT语音流畅性测试的音频，默认为None，此时程序根据转录文本自动判断
        """
        self.f_audio = input_f_audio
        self.f_trans = input_f_trans
        self.text = ''
        if self.f_trans.endswith('.txt'):
            try:
                with open(self.f_trans, "r", encoding="utf-8") as f:
                    self.text = f.read()
            except UnicodeDecodeError:
                with open(self.f_trans, "r", encoding="gb18030") as f:
                    self.text = f.read()
        text_no_punct = delete_punctuation(self.text)  # 删除标点符号后的文本
        text_seg_list_no_punct = delete_punctuation(jieba.lcut(self.text))  # 删除标点符号后的分词结果
        if is_pf is None:  # 若无显示说明输入的音频是语音流畅性PF还是语义流畅性SF，则自动判断
            has_pf = []
            for i_seg in text_seg_list_no_punct:
                if u'水' in i_seg:
                    has_pf.append(i_seg)
            if len(has_pf) / len(text_seg_list_no_punct) > 0.5:  # 分词结果中包含“水”字比例大于50%则判断为PF
                self.is_pf = True
            else:
                self.is_pf = False
        else:
            self.is_pf = is_pf
        # 语义流畅SF中排除的动物词汇前缀或后缀（为了统一名称，有效计算重复词汇），如大狗/小狗/狮子/豹子中的大/小/子
        self.ex_sf_prefix = [u'大', u'小', u'老', u'白', u'黑', u'黄', u'红', u'花', u'子', u'青', u'蓝',
                             u'肥', u'胖',
                             u'丑', u'寅', u'卯', u'辰', u'巳', u'午', u'未', u'申', u'酉', u'戌', u'亥', ]
        self.miss_w_dir = miss_w_dir
        self.ag_dict = audio_word_align(self.f_audio, text_no_punct, temp_dir, miss_w_dir=miss_w_dir)
        self.func_check_align_miss_words()
        self.cor_word = self.func_correct_word()  # 针对FT的转录文本，修正词语列表

    def func_check_align_miss_words(self):
        """
        对对齐的缺失汉字进行检查，作为后续VFT特征的前提，以便进行自动纠正和避免后续特征提取出错
        :return: None
        """
        miss_wd_all = []
        # 记录对齐时非英文的缺失汉字
        with open(self.miss_w_dir + '/miss_wrd_align.txt', 'r', encoding='utf-8') as f:
            for i_line in f.readlines():
                if zhong.match(i_line.strip()):
                    miss_wd_all.append(i_line.strip())
        # 检查这些缺失汉字拼音是否存在于字典，若存在，则可以用已存在的相同读音的汉字添加至字典
        with open(self.miss_w_dir + '/miss_wrd_align_cancor.log', 'a+', encoding='utf-8') as f:
            f.seek(0)
            if not f.readlines():
                f.write(f'\n----------- 以下缺失汉字可以自动校正，请直接复制至文件"{P2FA_MODEL}/dict -----------\n\n')
            f.seek(0)
            has = []
            for i_wd in f.readlines():
                has.append(i_wd.split('  ')[0])
            for mis_wd in miss_wd_all:
                mis_wd_py = ' '.join(pypinyin.lazy_pinyin(mis_wd, style=pypinyin.Style.TONE3))
                if (mis_wd_py in ALIGN.keys()) and (mis_wd not in has):
                    f.write(mis_wd + ALIGN[mis_wd_py] + '\n')
        # 检查这些缺失汉字拼音是否存在于字典，若不存在，记录下来，待后续手工修改
        with open(self.miss_w_dir + '/miss_wrd_align_nocor.log', 'a+', encoding='utf-8') as f:
            f.seek(0)
            if not f.readlines():
                f.write(f'\n----------- 以下缺失汉字无法自动校正，请在文件"{P2FA_MODEL}/dict中手动修改 -----------\n\n')
            f.seek(0)
            has = []
            for i_wd in f.readlines():
                has.append(i_wd.strip())
            for mis_wd in miss_wd_all:
                if (' '.join(pypinyin.lazy_pinyin(mis_wd, style=pypinyin.Style.TONE3)) not in ALIGN.keys()) \
                        and (mis_wd not in has):
                    f.write(mis_wd + '\n')
        if miss_wd_all:
            raise InterruptedError(f'音频-文本对齐过程中，P2FA的字典文件中缺失汉字: {miss_wd_all}（详见miss_wrd_align.txt文件），\n'
                                   f'请先在文件"{P2FA_MODEL}/dict"中补全，\n接着删除miss_wrd_align.txt文件，再运行！')

    def func_correct_word(self):
        """
        根据原始的转录文本（PF:删除标点符号后的文本，SF：删除标点符号后的分词结果），校正修正词语列表
        :return: 校正后的词语列表
        """
        cor_wd = []  # 校正后的词汇列表
        filter_punt = lambda s: ''.join(filter(lambda x: x not in STOPWORDS, s))  # 去除停用词
        text_no_punstop = ''.join(regex.findall(r'[\u4e00-\u9fff]+',
                                                delete_punctuation(filter_punt(self.text))))  # 删除标点符号/非中文字符后的无停用词文本
        seg_no_punstop = delete_punctuation(jieba.lcut(filter_punt(self.text)))  # 删除标点符号后的无停用词分词结果
        if self.is_pf:
            # 校正“水”开头词语/句子：该项校正的前提是确保所说的词汇列表均为“水”字开头
            for i_wd in list(filter(None, text_no_punstop.split(u'水'))):  # 按“水”字分割，去除空元素
                cor_wd.append(u'水' + i_wd)  # 添加每个词语开头的“水”字
        else:
            # 校正动物名称：若词汇或对应拼音在动物字典中存在，则原样保留或对应拼音修正，否则记录在miss_wrd_ani.txt，
            # 先从分词之后的词汇开始，搜寻不到便拆成单个字再进行匹配，英文字符直接记录在miss_wrd_ani.txt
            # 可以在每次运行完之后，检查miss_wrd_ani.txt并手动修正转录文本或在动物词典中添加新词
            for i_seg in seg_no_punstop:
                if i_seg in ANIMAL.keys():  # 字典中存在，则直接保存输出
                    cor_wd.append(i_seg)
                else:  # 若不存在
                    if not zhong.match(i_seg):  # 为非中文字符，跳过
                        with open(self.miss_w_dir + '/miss_wrd_ani.txt', 'a', encoding='utf-8') as f:
                            f.write(os.path.basename(self.f_trans) + ': ' + i_seg + '\n')
                        continue
                    i_seg_py = ' '.join(pypinyin.lazy_pinyin(i_seg, pypinyin.Style.TONE3))
                    if i_seg_py in ANIMAL.values():  # 当对应的拼音存在时，校正为字典中对应汉字词汇
                        i_seg_cor = list(ANIMAL.keys())[list(ANIMAL.values()).index(i_seg_py)]
                        cor_wd.append(i_seg_cor)
                        with open(self.miss_w_dir + '/miss_wrd_ani.txt', 'a', encoding='utf-8') as f:
                            f.write(os.path.basename(self.f_trans) + ': ' + i_seg + f'(已修正为{i_seg_cor})\n')
                    else:  # 当对应的拼音也不存在时，开始分解中文字符，尝试在字符中寻找动物词汇
                        for j_wd in i_seg:
                            if j_wd in ANIMAL.keys():
                                cor_wd.append(j_wd)
                            else:
                                if not zhong.match(j_wd):  # 为非中文字符，跳过
                                    with open(self.miss_w_dir + '/miss_wrd_ani.txt', 'a', encoding='utf-8') as f:
                                        f.write(os.path.basename(self.f_trans) + ': ' + j_wd + '\n')
                                    continue
                                j_wd_py = pypinyin.pinyin(j_wd, pypinyin.Style.TONE3, heteronym=True)[0]  # 单字考虑多音字
                                for k_py in range(len(j_wd_py)):
                                    if j_wd_py[k_py] in ANIMAL.values():  # 当对应的拼音存在时，校正为字典中对应汉字词汇
                                        j_wd_cor = list(ANIMAL.keys())[list(ANIMAL.values()).index(j_wd_py[k_py])]
                                        cor_wd.append(j_wd_cor)
                                        with open(self.miss_w_dir + '/miss_wrd_ani.txt', 'a', encoding='utf-8') as f:
                                            f.write(os.path.basename(self.f_trans) + ': '
                                                    + j_wd + f'(已修正为{j_wd_cor})\n')
                                        break
                                    elif k_py == (len(j_wd_py) - 1):
                                        cor_wd.append(j_wd)
                                        with open(self.miss_w_dir + '/miss_wrd_ani.txt', 'a', encoding='utf-8') as f:
                                            if j_wd not in self.ex_sf_prefix:
                                                f.write(os.path.basename(self.f_trans) + ': ' + j_wd + '\n')
        return cor_wd

    def func_item_all(self):
        """
        计算整个任务尺度的“水”开头或动物词的全部词语列表，包括重复的词语
        :return: 全部词语列表，包括重复的词语
        """
        if self.is_pf:
            return self.func_correct_word()
        else:  # 针对SF动物词汇排除一些前缀或后缀
            filter_wd = lambda s: ''.join(filter(lambda x: x not in self.ex_sf_prefix, s))
            cor_word = []
            for i_wd in self.cor_word:
                cor_word.append(filter_wd(i_wd))
            cor_word = list(filter(None, cor_word))
            return cor_word

    def func_item_bin(self):
        """
        获取每10秒尺度（对应为一个bin）的“水”开头或动物词和时间对应列表
        :return: 每10秒间隔中的非重复的词语列表，任务总时长为30秒，则这里返回1*3列表，
        其中每个元素对应每个bin的词语-时间列表，即格式为[[(词语a1, (开始时间，结束时间)), (词语a2, (开始时间，结束时间)), ...],
                                                [(词语b1, (开始时间，结束时间)), (词语b2, (开始时间，结束时间)), ...], ...]
        """
        wd_time_bin1, wd_time_bin2, wd_time_bin3 = [], [], []
        ag_dict = self.ag_dict.copy()
        for key in self.ag_dict:  # 对齐文本删除停用词
            if self.ag_dict[key] in STOPWORDS:
                ag_dict.pop(key)
        if self.is_pf:  # PFT需删除连续说“水”的字，一个位置仅保留最后一个“水”字
            dump = None
            for key in self.ag_dict:
                if (dump == self.ag_dict[key]) and (dump == u'水'):
                    ag_dict.pop(list(self.ag_dict.keys())[list(self.ag_dict.keys()).index(key) - 1])
                dump = self.ag_dict[key]
            if list(self.ag_dict.values())[-1] == u'水':  # 若最后一个为“水”，则删除
                ag_dict.popitem()
        else:  # SFT需要删除对应的需要排除的动物词汇中指定的前缀或后缀
            for key in self.ag_dict:
                if self.ag_dict[key] in self.ex_sf_prefix:
                    ag_dict.pop(key)
        wd_no_rep = []
        for i_wd in self.func_item_all():
            if i_wd not in wd_no_rep:  # 仅保留非重复词语
                wd_ts = list(ag_dict.keys())[0][0]
                wd_te = list(ag_dict.keys())[len(i_wd) - 1][-1]
                wd_no_rep.append(i_wd)
                if wd_te <= 10.0:
                    wd_time_bin1.append((i_wd, (wd_ts, wd_te)))
                elif wd_te <= 20.0:
                    wd_time_bin2.append((i_wd, (wd_ts, wd_te)))
                else:
                    wd_time_bin3.append((i_wd, (wd_ts, wd_te)))
            for j in range(len(i_wd)):
                ag_dict.pop(list(ag_dict.keys())[0])
        return [wd_time_bin1, wd_time_bin2, wd_time_bin3]

    def func_clusters(self):
        """
        根据语义距离进行聚类：以所有词语之间的语义距离的均值作为分割不同语义簇的语义阈值（包括重复的词语），
        若前后连续词间的语义距离大于该阈值，则属于不同的簇，否则属于同一簇。嵌入为300d的fastText中的中文词向量预训练模型
        该特征多数只对语音流畅性测试SFT进行，对于语音流畅性PFT这里同样计算，不过有的通过临床上的音素簇来计算，
        见Troyer,1997，Clustering and Switching as Two Components of Verbal Fluency:
        Evidence From Younger and Older Healthy Adults
        :return: 语义簇二维列表，其中每个元素对应每个簇，每个簇包含属于该簇的所有词语列表
        """
        sd_all = []
        for i_wd_index in range(len(self.func_item_all())):
            for j_wd_index in range(len(self.func_item_all())):
                if i_wd_index == j_wd_index:
                    continue
                i_wd, j_wd = self.func_item_all()[i_wd_index], self.func_item_all()[j_wd_index]
                s1 = MODEL_fastText[i_wd]
                s2 = MODEL_fastText[j_wd]
                if self.is_pf and len(i_wd) > 4:  # 针对PFT中说句子的情形
                    s1 = MODEL_fastText.get_sentence_vector(' '.join(jieba.cut(i_wd)))
                elif self.is_pf and len(j_wd) > 4:
                    s2 = MODEL_fastText.get_sentence_vector(' '.join(jieba.cut(j_wd)))
                sd_all.append(distance.cosine(s1, s2))
        sd_thr = np.mean(sd_all)  # 全部两两词语之间的语义距离的均值作为分割不同语义簇的语义阈值
        clusters = [[self.func_item_all()[0]]]
        clu_index = 0
        for i_wd_index in range(len(self.func_item_all())):
            if i_wd_index < len(self.func_item_all()) - 1:
                i_wd, j_wd = self.func_item_all()[i_wd_index], self.func_item_all()[i_wd_index + 1]
                s1 = MODEL_fastText[i_wd]
                s2 = MODEL_fastText[j_wd]
                if self.is_pf and len(i_wd) > 4:  # 针对PFT中说句子的情形
                    s1 = MODEL_fastText.get_sentence_vector(' '.join(jieba.cut(i_wd)))
                elif self.is_pf and len(j_wd) > 4:
                    s2 = MODEL_fastText.get_sentence_vector(' '.join(jieba.cut(j_wd)))
                cur_sd = distance.cosine(s1, s2)
                if cur_sd <= sd_thr:
                    clusters[clu_index].append(j_wd)
                else:
                    clu_index += 1
                    clusters.append([j_wd])
        return clusters

    def item_num_no_rep(self):
        """
        计算整个任务尺度的“水”开头或动物词的词语数量，不包括重复的词语
        :return: 词语数量，不包括重复的词语
        """
        wd_num_no_rep = len(set(self.func_item_all()))
        return wd_num_no_rep

    def item_num_rep(self):
        """
        计算整个任务尺度的“水”开头或动物词，重复的词语数量，也称为Perseverations（持续言语）
        :return: 重复的词语数量
        """
        wd_num_rep = len(self.func_item_all()) - len(set(self.func_item_all()))
        return wd_num_rep

    def item_num_intrusions(self):
        """
        计算不属于目标词汇的数量，即对于PFT为非“水”字开头的词语数量，对于SFT为非动物词汇的数量，也称为Intrusions
        对于SFT，需要确保动物词汇列表ANIMAL中齐全，检查miss_wrd_ani.txt并手动修正转录文本或在动物词典中添加新词
        对于PFT，由于前面的校正前提是确保所说的词汇列表均为“水”字开头，因此这里计算出的Intrusions恒等于0，所以这里不适合计算PFT的该特征
        :return: 非目标词汇的数量
        """
        intrusions = 0
        for i_wd in self.func_item_all():
            if self.is_pf:
                # 由于前面校正，这里恒不成立，所以计算出的Intrusions恒等于0（这里不适合计算PFT的该特征，后面要根据手工转录计算）
                if not i_wd.startswith(u'水'):
                    intrusions += 1
            else:
                if i_wd not in ANIMAL.keys():
                    intrusions += 1
        return intrusions

    def item_num_no_rep_bin(self):
        """
        计算每10秒尺度（对应为一个bin）的“水”开头或动物词的数量，排除重复的词语
        :return: 每10秒间隔中的非重复的词语数量，任务总时长为30秒，则这里返回1*3列表，其中每个元素对应每个bin的词语数量
        """
        wd_num_bin = [len(i) for i in self.func_item_bin()]
        return wd_num_bin

    def temporal_distance_bin(self):
        """
        计算每10秒尺度（对应为一个bin）的“水”开头或动物词的平均时间距离，排除重复的词语：当前词结束与下一词开始之间的转换时间，每个bin内作平均处理
        :return: 每10秒间隔中的非重复的词语的平均时间距离列表，任务总时长为30秒，则这里返回1*3列表，其中每个元素对应每个bin的平均时间距离
        """
        td = []
        for i_bin in self.func_item_bin():
            td_bin = []
            for j in range(len(i_bin)):
                if j < len(i_bin) - 1:
                    td_bin.append(round(i_bin[j + 1][1][0] - i_bin[j][1][-1], 4))
            if td_bin:  # 当前bin有多余一个词语元素
                td.append(td_bin)
            else:  # 当前bin仅有一个或者无词语元素
                # 当前bin仅有一个词语元素：时间距离定义为当前bin数*bin长10秒-该词语元素的结束时间，
                # 即假定理想情况下，当前bin刚结束被试便说出了下一个有效词语
                if len(i_bin):
                    td.append([round(10.0 * (self.func_item_bin().index(i_bin) + 1) - i_bin[-1][1][-1], 4)])
                else:  # 当前bin无词语元素：时间距离定义为10s，也即假定理想情况下，被试在当前bin开始和结束时刻各说出一个有效词汇
                    td.append([10.0])
        td_mean_bin = [round(np.mean(i), 4) for i in td]
        return td_mean_bin

    def semantic_distance_bin(self):
        """
        计算每10秒尺度（对应为一个bin）的“水”开头或动物词的平均语义距离，排除重复的词语：
        定义为词向量间的余弦距离，即1-当前词与下一词间的余弦相似性，每个bin内作平均处理。
        词向量嵌入采用fastText中的中文预训练模型：300d
        :return: 每10秒间隔中的非重复的词语的平均语义距离列表，任务总时长为30秒，则这里返回1*3列表，其中每个元素对应每个bin的平均语义距离
        """
        sd = []
        for i_bin in self.func_item_bin():
            sd_bin = []
            for j in range(len(i_bin)):
                if j < len(i_bin) - 1:
                    s1 = MODEL_fastText[i_bin[j][0]]
                    s2 = MODEL_fastText[i_bin[j + 1][0]]
                    if self.is_pf and len(i_bin[j][0]) > 4:  # 针对PFT中说句子的情形，假定超过4字为句子
                        s1 = MODEL_fastText.get_sentence_vector(' '.join(jieba.cut(i_bin[j][0])))
                    elif self.is_pf and len(i_bin[j + 1][0]) > 4:
                        s2 = MODEL_fastText.get_sentence_vector(' '.join(jieba.cut(i_bin[j + 1][0])))
                    sd_bin.append(distance.cosine(s1, s2))
            if sd_bin:  # 当前bin有多余一个词语元素
                sd.append(sd_bin)
            else:  # 当前bin仅有一个或者无词语元素：语义距离定义为假定当前bin内说出了两个相同的词，即为0
                sd.append([0.0])
        sd_mean_bin = [round(np.mean(i), 4) for i in sd]
        return sd_mean_bin

    def levenshtein_distance_bin(self):
        """
        计算每10秒尺度（对应为一个bin）的“水”开头或动物词的平均编辑距离（即Levenshtein距离，评价字符串间的相似性），排除重复的词语，每个bin内作平均处理：
        描述由一个字串转化成另一个字串最少的操作次数，其中操作包括插入insertions、删除deletions、替换substitutions
        :return: 每10秒间隔中的非重复的词语的平均编辑距离列表，任务总时长为30秒，则这里返回1*3列表，其中每个元素对应每个bin的平均编辑距离
        """
        epi = epitran.Epitran(code='cmn-Hans', cedict_file=CEDICT, tones=True)  # 转换成国际音标IPA来进行距离度量
        ld = []
        for i_bin in self.func_item_bin():
            ld_bin = []
            for j in range(len(i_bin)):
                if j < len(i_bin) - 1:
                    s1 = epi.transliterate(i_bin[j][0])
                    s2 = epi.transliterate(i_bin[j + 1][0])
                    ld_bin.append(editdistance.eval(s1, s2))
            if ld_bin:  # 当前bin有多余一个词语元素
                ld.append(ld_bin)
            else:  # 当前bin仅有一个或者无词语元素：编辑距离定义为假定当前bin内说出了两个相同的词，即为0
                ld.append([0.0])
        sd_mean_bin = [np.mean(i) for i in ld]
        return sd_mean_bin

    def cluster_size(self):
        """
        计算语义簇的平均词语数量（簇大小）：语义集群中所有词汇数量的平均值
        该特征多数只对语音流畅性测试SFT进行，对于语音流畅性PFT这里同样计算，不过有的通过临床上的音素簇来计算，
        见Troyer,1997，Clustering and Switching as Two Components of Verbal Fluency:
        Evidence From Younger and Older Healthy Adults
        :return: 语义簇的平均词语数量
        """
        clusters = self.func_clusters()
        clu_size = np.mean([len(i) for i in clusters])
        return clu_size

    def cluster_switches(self):
        """
        计算语义簇之间的切换次数：簇间切换的总次数（从一个簇过渡到另一个簇记为一次切换）
        该特征多数只对语音流畅性测试SFT进行，对于语音流畅性PFT这里同样计算，不过有的通过临床上的音素簇来计算，
        见Troyer,1997，Clustering and Switching as Two Components of Verbal Fluency:
        Evidence From Younger and Older Healthy Adults
        :return: 语义簇间的总切换次数
        """
        clu_sw = len(self.func_clusters()) - 1
        return clu_sw

    def word_frequency(self):
        """
        根据现代汉语语料库分词类词频表，计算词汇的平均词频，排除重复词汇
        :return: 平均词频，不包括重复词汇
        """
        wd_wf = []
        for i_wd in set(self.func_item_all()):
            if self.is_pf and len(i_wd) > 4:  # 针对PFT中说句子的情形
                j_seg = []
                for j_wd in jieba.lcut(i_wd):  # 分割计算
                    if j_wd in CWF.keys():
                        j_seg.append(CWF[j_wd])
                    else:
                        j_seg.append(0.0)
                wd_wf.append(np.mean(j_seg))
            else:
                if i_wd in CWF.keys():
                    wd_wf.append(CWF[i_wd])
                else:
                    wd_wf.append(0.0)
        wf = np.mean(wd_wf)
        return wf

    def get_all_feat(self, prefix='VFT_', use_=False):
        """
        获取当前所有特征
        :param prefix: pd.DataFrame类型特征列名的前缀
        :param use_: 是否提取包含分类贡献度低、ICC低的（方法有_标记）特征，默认否
        :return: 该类的全部特征, pd.DataFrame类型
        """
        num_nr = self.item_num_no_rep()
        num_r = self.item_num_rep()
        num_intr = self.item_num_intrusions()
        num_nr_bin = self.item_num_no_rep_bin()
        td_bin = self.temporal_distance_bin()
        sd_bin = self.semantic_distance_bin()
        ld_bin = self.levenshtein_distance_bin()
        cz = self.cluster_size()
        csw = self.cluster_switches()
        wf = self.word_frequency()
        if use_:
            feat = {prefix+"Word Count": [num_nr], prefix+"Repetitions": [num_r], prefix+"Intrusions": [num_intr],
                    prefix+"WC Bin1": [num_nr_bin[0]], prefix+"WC Bin2": [num_nr_bin[1]], prefix+"WC Bin3": [num_nr_bin[2]],
                    prefix+"TD Bin1": [td_bin[0]], prefix+"TD Bin2": [td_bin[1]], prefix+"TD Bin3": [td_bin[2]],
                    prefix+"SD Bin1": [sd_bin[0]], prefix+"SD Bin2": [sd_bin[1]], prefix+"SD Bin3": [sd_bin[2]],
                    prefix+"LD Bin1": [ld_bin[0]], prefix+"LD Bin2": [ld_bin[1]], prefix+"LD Bin3": [ld_bin[2]],
                    prefix+"Cluster Size": [cz], prefix+"Cluster Switches": [csw],
                    prefix+"Word Frequency(%)": [wf], }
        else:
            feat = {prefix+"Word Count": [num_nr], prefix+"Repetitions": [num_r], prefix+"Intrusions": [num_intr],
                    prefix+"WC Bin1": [num_nr_bin[0]], prefix+"WC Bin2": [num_nr_bin[1]], prefix+"WC Bin3": [num_nr_bin[2]],
                    prefix+"TD Bin1": [td_bin[0]], prefix+"TD Bin2": [td_bin[1]], prefix+"TD Bin3": [td_bin[2]],
                    prefix+"SD Bin1": [sd_bin[0]], prefix+"SD Bin2": [sd_bin[1]], prefix+"SD Bin3": [sd_bin[2]],
                    prefix+"LD Bin1": [ld_bin[0]], prefix+"LD Bin2": [ld_bin[1]], prefix+"LD Bin3": [ld_bin[2]],
                    prefix+"Cluster Size": [cz], prefix+"Cluster Switches": [csw],
                    prefix+"Word Frequency(%)": [wf], }
        return pd.DataFrame(feat)


if __name__ == "__main__":
    sid = os.path.basename(DATA_PATH_EXAMPLE).split('_')[-1]
    temp_path = os.path.join(parent_path, 'results/temp')
    miss_w_path = os.path.join(parent_path, 'results/missing_words')
    audio_file_p = os.path.join(DATA_PATH_EXAMPLE, f"session_1/05_VFT/phonetic/{sid}_phonetic.wav")
    audio_file_s = os.path.join(DATA_PATH_EXAMPLE, f"session_1/05_VFT/semantic/{sid}_semantic.wav")
    trans_file_p = os.path.join(TRANS_PATH_EXAMPLE, f"session_1/05_VFT/phonetic/{sid}_phonetic.txt")
    trans_file_s = os.path.join(TRANS_PATH_EXAMPLE, f"session_1/05_VFT/semantic/{sid}_semantic.txt")
    vft_feat = VFTFeatures(audio_file_s, trans_file_s, temp_path, miss_w_dir=miss_w_path).get_all_feat()
    print(vft_feat)

    # 总体运行总的特征提取程序，比如main.py时，先提前运行以下，进行对齐字典缺失词的填补和动物词的填补、修正等
    # import os, glob
    # trans_data_dir = os.path.join(TRANS_PATH, 'ASR')
    # subject_dir_list = []  # 每个被试的音频主文件路径组成的列表
    # for i_each_file in os.listdir(os.path.join(grandpa_path, 'data')):
    #     data_path_group = os.path.join(os.path.join(grandpa_path, 'data'), i_each_file)
    #     for j_each_file in os.listdir(data_path_group):
    #         data_path_gender = os.path.join(data_path_group, j_each_file)
    #         for k_each_file in os.listdir(data_path_gender):
    #             data_path_sess = os.path.join(data_path_gender, k_each_file)
    #             for l_each_file in os.listdir(data_path_sess):
    #                 subject_dir_list.append(os.path.join(data_path_sess, l_each_file))
    # for subject_dir in subject_dir_list:
    #     print("---------- Processing %d / %d: %s ----------" %
    #           (subject_dir_list.index(subject_dir) + 1, len(subject_dir_list), subject_dir))
    #     subject_id = os.path.normpath(subject_dir).split(os.sep)[-2].split("_")[1]
    #     csv_data = read_csv(subject_dir + "/" + subject_id + ".csv")
    #     sess = int(csv_data[0][4].split("：")[1])
    #     pft_audio = os.path.join(subject_dir, "05_VFT/phonetic", subject_id + "_phonetic.wav")
    #     sft_audio = os.path.join(subject_dir, "05_VFT/semantic", subject_id + "_semantic.wav")
    #     if os.path.exists(pft_audio):
    #         try:
    #             pft_trans = glob.glob(os.path.join(trans_data_dir,
    #                                                f'**/session_{sess}/**/{subject_id}_phonetic.txt'),
    #                                   recursive=True)[0]
    #         except IndexError:
    #             raise FileNotFoundError(f'session_{sess}/**/{subject_id}_phonetic.txt 不存在')
    #         pft_feat = VFTFeatures(pft_audio, pft_trans, temp_path, miss_w_path, True)
    #     if os.path.exists(sft_audio):
    #         try:
    #             sft_trans = glob.glob(os.path.join(trans_data_dir,
    #                                                f'**/session_{sess}/**/{subject_id}_semantic.txt'),
    #                                   recursive=True)[0]
    #         except IndexError:
    #             raise FileNotFoundError(f'session_{sess}/**/{subject_id}_semantic.txt 不存在')
    #         sft_feat = VFTFeatures(sft_audio, sft_trans, temp_path, miss_w_path, False)
