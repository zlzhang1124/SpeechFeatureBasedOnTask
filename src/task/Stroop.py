#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/3/28 16:21 
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : Stroop.py
# @Software : Python3.6; PyCharm; Windows10 / Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M / 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090
# @Version  : V1.1.0 - ZL.Z：2022/5/19
# 		      使用To TextGrid (silences)方法替换To TextGrid (vuv)，仅分割有声和无声段
#             V1.0 - ZL.Z：2022/3/28 - 2022/3/29
# 		      First version.
# @License  : None
# @Brief    : 色词测试任务相关特征：Time(s)/Correct Count/SIE Time(s)/SIE Count

from src.utils.util import *
from src.config import *
import glob
import regex
import string
import numpy as np
import pypinyin

STOP_WORDS = os.path.join(parent_path, "dicts/stop_words_vft.txt")
STOPWORDS = [line.strip() for line in open(STOP_WORDS, encoding='utf-8').readlines()]


class StroopFeatures:
	"""获取Stroop任务的特征"""
	def __init__(self, input_dir_audio: str, input_dir_trans: str, f0min=75, f0max=600):
		"""
		初始化
		:param input_dir_audio: 输入包含该任务中三个.wav音频文件的文件夹
		:param input_dir_trans: 输入包含该任务中三个.txt文本转录文件的文件夹
		:param f0min: 最小追踪pitch,默认75Hz
        :param f0max: 最大追踪pitch,默认600Hz
		"""
		self.f0min = f0min
		self.f0max = f0max
		self.has_dot, self.has_hz, self.has_stt = True, True, True  # 默认三个音频文件均存在
		audios, trans_text = [], []
		for suffix in 'dots', 'hanzi', 'STT':
			audio_l = glob.glob(os.path.join(input_dir_audio, f'*_{suffix}.wav'))
			if audio_l:
				audios.append(audio_l[0])
				try:
					trans = glob.glob(os.path.join(input_dir_trans, f'*_{suffix}.txt'))[0]
					try:
						with open(trans, "r", encoding="utf-8") as f:
							trans_text.append(f.read())
					except UnicodeDecodeError:
						with open(trans, "r", encoding="gb18030") as f:
							trans_text.append(f.read())
				except IndexError:
					raise FileNotFoundError(f'{input_dir_trans}/*_{suffix}.txt 不存在，但对应音频存在')
			else:
				audios.append('')
				trans_text.append('')
				if suffix == 'dots':
					self.has_dot = False
				if suffix == 'hanzi':
					self.has_hz = False
				if suffix == 'STT':
					self.has_stt = False
		self.aud_dot, self.aud_hz, self.aud_stt = audios[0], audios[1], audios[2]
		self.text_dot, self.text_hz, self.text_stt = trans_text[0], trans_text[1], trans_text[2]
		filter_punt = lambda s: ''.join(filter(lambda x: x not in STOPWORDS, s))  # 去除停用词
		self.dot_clr = ''.join(regex.findall(r'[\u4e00-\u9fff]+',
		                                     delete_punctuation(filter_punt(self.text_dot))))  # 删除标点符号/非中文字符后的无停用词文本
		self.hz_clr = ''.join(regex.findall(r'[\u4e00-\u9fff]+',
		                                    delete_punctuation(filter_punt(self.text_hz))))
		self.stt_clr = ''.join(regex.findall(r'[\u4e00-\u9fff]+',
		                                     delete_punctuation(filter_punt(self.text_stt))))
		self.dot_clr, self.hz_clr, self.stt_clr = self.func_correct_word()  # 针对转录文本，修正词语后的每四字组成列表
		self.cor_dot = [[u'红', u'黄', u'蓝', u'绿'], [u'绿', u'蓝', u'黄', u'红'], [u'黄', u'红', u'绿', u'蓝']]
		self.cor_hz = [[u'黄', u'红', u'绿', u'蓝'], [u'绿', u'黄', u'蓝', u'红'], [u'蓝', u'绿', u'红', u'黄']]
		self.cor_stt = [[u'黄', u'红', u'绿', u'蓝'], [u'绿', u'蓝', u'黄', u'红'], [u'红', u'蓝', u'绿', u'黄']]

	def func_correct_word(self):
		"""
		校正转录的文本
		:return: 每个任务校正后的转录文本对应的每四字（即一行）列表组成的元组：(self.dot_clr、self.hz_clr、self.stt_clr)
		"""
		allow_wd = [u'红', u'黄', u'蓝', u'绿']
		allow_pinyin = {'hong': u'红', 'huang': u'黄', 'lan': u'蓝', 'lv': u'绿'}
		dot_hz_stt = []
		for item in self.dot_clr, self.hz_clr, self.stt_clr:
			wds = []
			if len(item) == 12:  # 当文本数量满足要求，即12个颜色字
				for i_wd in item:
					i_wd_py = pypinyin.lazy_pinyin(i_wd, pypinyin.Style.TONE3)[0].strip(string.digits)
					if i_wd in allow_wd:  # 正确转录的颜色字
						wds.append(i_wd)
					elif i_wd_py in allow_pinyin.keys():  # 错误的转录，但拼音正确
						wds.append(allow_pinyin[i_wd_py])
			else:  # 当文字数量不满足，直接判断字是否相同
				for i_wd in item:
					if len(wds) >= 12:  # 仅保留前12个汉字
						break
					if i_wd in allow_wd:  # 正确转录的颜色字
						wds.append(i_wd)
			if len(wds) < 12:  # 确保总长度为12个汉字，不足则用np.nan补齐
				for num in range(12 - len(wds)):
					wds.append(np.nan)
			dot_hz_stt.append([wds[i:i + 4] for i in range(0, len(wds), 4)])  # 每四个汉字分割成一维列表
		return tuple(dot_hz_stt)

	def total_time_use(self):
		"""
		获取三个任务音频的总用时时间，包括读带有颜色的圆点、读汉字、读带有不匹配颜色的汉字
		:return: 总用时，单位s，仅当三个音频均存在才返回，否则返回np.nan
		"""
		if self.has_dot and self.has_hz and self.has_stt:
			time_all = 0
			for aud in self.aud_dot, self.aud_hz, self.aud_stt:
				sd_obj = parselmouth.Sound(aud)
				text_grid_vuv = call(sd_obj, "To TextGrid (silences)", 100, 0.0, -25.0, 0.1, 0.1, 'U', 'V')
				vuv_info = call(text_grid_vuv, "List", 'no', 6, 'no', 'no')
				segments_v, segments_u = duration_from_vuvInfo(vuv_info)
				if len(segments_v):  # 为浊音
					time_len = segments_v[-1][-1] - segments_v[0][0]
				else:  # 为清音
					time_len = segments_u[-1][-1] - segments_u[0][0]
				time_all += time_len
			return time_all
		else:
			return np.nan

	def total_num_correct(self):
		"""
		获取三个任务的总正确数，包括读带有颜色的圆点、读汉字、读带有不匹配颜色的汉字
		:return: 总正确数，仅当三个转录文本均存在才返回，否则返回np.nan
		"""
		if self.has_dot and self.has_hz and self.has_stt:
			num_cor = 0
			for i_item in ([self.dot_clr, self.cor_dot], [self.hz_clr, self.cor_hz], [self.stt_clr, self.cor_stt]):
				num_cor += (np.array(i_item[0]) == np.array(i_item[1])).sum()
			return num_cor
		else:
			return np.nan

	def stroop_interference_effect_time(self):
		"""
		计算色词干扰效应SIE中的耗时数：色词STT任务的用时-汉字hanzi任务的用时
		:return: SIE耗时数，单位s，仅当色词和汉字任务音频均存在才返回，否则返回np.nan
		"""
		if self.has_hz and self.has_stt:
			sie_times = []
			for aud in self.aud_hz, self.aud_stt:
				sd_obj = parselmouth.Sound(aud)
				text_grid_vuv = call(sd_obj, "To TextGrid (silences)", 100, 0.0, -25.0, 0.1, 0.1, 'U', 'V')
				vuv_info = call(text_grid_vuv, "List", 'no', 6, 'no', 'no')
				segments_v, segments_u = duration_from_vuvInfo(vuv_info)
				if len(segments_v):  # 为浊音
					sie_times.append(segments_v[-1][-1] - segments_v[0][0])
				else:  # 为清音
					sie_times.append(segments_u[-1][-1] - segments_u[0][0])
			sie_time = sie_times[1] - sie_times[0]
			return sie_time
		else:
			return np.nan

	def stroop_interference_effect_num(self):
		"""
		计算色词干扰效应SIE中的正确数：色词STT任务的正确数-汉字hanzi任务的正确数
		:return: SIE正确数，仅当色词和汉字任务文本均存在才返回，否则返回np.nan
		"""
		if self.has_hz and self.has_stt:
			sie_num = (np.array(self.stt_clr) == np.array(self.cor_stt)).sum() - \
			          (np.array(self.hz_clr) == np.array(self.cor_hz)).sum()
			return sie_num
		else:
			return np.nan

	def get_all_feat(self, prefix='Stroop_', use_=False):
		"""
		获取当前所有特征
		:param prefix: pd.DataFrame类型特征列名的前缀
		:param use_: 是否提取包含分类贡献度低、ICC低的（方法有_标记）特征，默认否
		:return: 该类的全部特征, pd.DataFrame类型
		"""
		time_use = self.total_time_use()
		num_cor = self.total_num_correct()
		sie_time = self.stroop_interference_effect_time()
		sie_num = self.stroop_interference_effect_num()
		if use_:
			feat = {prefix+"Time(s)": [time_use], prefix+"Correct Count": [num_cor],
			        prefix+"SIE Time(s)": [sie_time], prefix+"SIE Count": [sie_num], }
		else:
			feat = {prefix+"Time(s)": [time_use], prefix+"Correct Count": [num_cor],
			        prefix+"SIE Time(s)": [sie_time], prefix+"SIE Count": [sie_num], }
		return pd.DataFrame(feat)


if __name__ == "__main__":
	audio_file_path = os.path.join(DATA_PATH_EXAMPLE, f"session_1/04_Stroop")
	trans_file_path = os.path.join(TRANS_PATH_EXAMPLE, f"session_1/04_Stroop")
	stroop_feat = StroopFeatures(audio_file_path, trans_file_path).get_all_feat()
	print(stroop_feat)

