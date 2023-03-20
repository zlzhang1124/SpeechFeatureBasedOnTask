#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/3/29 13:03
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : Naming.py
# @Software : Python3.6; PyCharm; Windows10 / Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M / 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090
# @Version  : V1.1 - ZL.Z：2022/5/19
# 		      获取时长方式改为直接获取音频长度
#             V1.0 - ZL.Z：2022/3/29
# 		      First version.
# @License  : None
# @Brief    : 图片命名测试任务相关特征：Time Mean(s)/Time SD(s)/Accuracy

from src.utils.util import *
from src.config import *
import regex
import numpy as np

STOP_WORDS = os.path.join(parent_path, "dicts/stop_words_vft.txt")
STOPWORDS = [line.strip() for line in open(STOP_WORDS, encoding='utf-8').readlines()]


class NamingFeatures:
	"""获取Naming任务的特征"""
	def __init__(self, input_dir_audio: str, input_dir_trans: str):
		"""
		初始化
		:param input_dir_audio: 输入包含该任务中.wav音频文件的文件夹
		:param input_dir_trans: 输入包含该任务中.txt文本转录文件的文件夹
		"""
		self.has_all = True  # 默认全部的30个音频文件均存在
		if len(os.listdir(input_dir_audio)) < 30:
			self.has_all = False
		self.audios, self.trans_text = [], []  # 所有音频/转录文本列表
		filter_punt = lambda s: ''.join(filter(lambda x: x not in STOPWORDS, s))  # 去除停用词
		for i_aud in os.listdir(input_dir_audio):
			if i_aud.endswith('.wav'):
				self.audios.append(os.path.join(input_dir_audio, i_aud))
				trans = os.path.join(input_dir_trans, i_aud.replace('.wav', '.txt'))
				try:
					with open(trans, "r", encoding="utf-8") as f:
						_text = f.read()
				except UnicodeDecodeError:
					with open(trans, "r", encoding="gb18030") as f:
						_text = f.read()
				except FileNotFoundError:
					raise FileNotFoundError(f'{trans} 不存在，但对应音频存在')
				text = ''.join(regex.findall(r'[\u4e00-\u9fff]+',
				                             delete_punctuation(filter_punt(_text))))  # 删除标点符号/非中文字符后的无停用词文本
				self.trans_text.append(text)
		self.cor_naming = {'T001': [u'筷子'], 'T002': [u'钥匙', u'门钥匙'], 'T003': [u'自行车', u'单车'],
		                   'T004': [u'袜子', u'长袜子', u'黑袜子'], 'T005': [u'香蕉'], 'T006': [u'水龙头'],
		                   'T007': [u'羽毛球'], 'T008': [u'手表', u'表'], 'T009': [u'遥控器', u'电视遥控器'],
		                   'T010': [u'茄子'], 'T011': [u'南瓜'], 'T012': [u'火车'], 'T013': [u'打火机'],
		                   'T014': [u'圆规'], 'T015': [u'地球仪'], 'T016': [u'钢琴'],
		                   'T017': [u'电热毯', u'电热毯子', u'电毯', u'电毯子'], 'T018': [u'望远镜'],
		                   'T019': [u'兔子', u'小兔子'], 'T020': [u'毛笔'], 'T021': [u'金鱼'],
		                   'T022': [u'电子琴', u'电子钢琴'], 'T023': [u'马'], 'T024': [u'蜻蜓'],
		                   'T025': [u'孔雀'], 'T026': [u'海星'], 'T027': [u'猫头鹰'], 'T028': [u'马车'],
		                   'T029': [u'袋鼠'], 'T030': [u'蜈蚣']}

	def time_use_mean(self):
		"""
		获取任务中全部30个命名音频的平均用时，即每张图片的平均命名时间
		:return: 平均命名时间，单位s，仅当30个音频均存在才返回，否则返回np.nan
		"""
		if self.has_all:
			times = []
			for aud in self.audios:
				sd_obj = parselmouth.Sound(aud)
				times.append(sd_obj.get_total_duration())
			return np.mean(times)
		else:
			return np.nan

	def time_use_sd(self):
		"""
		获取任务中全部30个命名音频的用时标准差，即每张图片的命名时间标准差
		:return: 命名时间标准差，单位s，仅当30个音频均存在才返回，否则返回np.nan
		"""
		if self.has_all:
			times = []
			for aud in self.audios:
				sd_obj = parselmouth.Sound(aud)
				times.append(sd_obj.get_total_duration())
			return np.std(times)
		else:
			return np.nan

	def accuracy(self):
		"""
		获取图片命名任务的准确率：命名正确数量/30
		:return: 正确率，仅当全部30个音频均存在才返回，否则返回np.nan
		"""
		if self.has_all:
			num_cor = 0
			for i_index in range(30):  # 共30张命名图片
				cor = self.cor_naming[os.path.basename(self.audios[i_index]).strip('.wav').split('_')[-1]]
				anw = self.trans_text[i_index]
				if anw in cor:  # 判断所答是否在正确答案列表中
					num_cor += 1
				# else:
				# 	print(os.path.basename(self.audios[i_index]), anw)
			return num_cor / 30
		else:
			return np.nan

	def get_all_feat(self, prefix='Naming_', use_=False):
		"""
		获取当前所有特征
		:param prefix: pd.DataFrame类型特征列名的前缀
		:param use_: 是否提取包含分类贡献度低、ICC低的（方法有_标记）特征，默认否
		:return: 该类的全部特征, pd.DataFrame类型
		"""
		time_mean = self.time_use_mean()
		time_sd = self.time_use_sd()
		acc = self.accuracy()
		if use_:
			feat = {prefix+"Time Mean(s)": [time_mean], prefix+"Time SD(s)": [time_sd],
			        prefix+"Accuracy": [acc], }
		else:
			feat = {prefix+"Time Mean(s)": [time_mean], prefix+"Time SD(s)": [time_sd],
			        prefix+"Accuracy": [acc], }
		return pd.DataFrame(feat)


if __name__ == "__main__":
	audio_file_path = os.path.join(DATA_PATH_EXAMPLE, f"session_1/06_Naming")
	trans_file_path = os.path.join(TRANS_PATH_EXAMPLE, f"session_1/06_Naming")
	naming_feat = NamingFeatures(audio_file_path, trans_file_path).get_all_feat()
	print(naming_feat)

