#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/3/1 9:48 
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : SRepeating.py
# @Software : Python3.6; PyCharm; Windows10 / Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M / 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090
# @Version  : V1.1.0 - ZL.Z：2022/5/19
# 		      使用To TextGrid (silences)方法替换To TextGrid (vuv)，仅分割有声和无声段
#             V1.0 - ZL.Z：2022/3/1
# 		      First version.
# @License  : None
# @Brief    : 提取SRepeating句子复述任务的特征：句子相似度

from dtaidistance import dtw_ndim
from src.utils.util import *
from src.config import *


def speech_similarity(audio_template: str, audio: str):
	"""
	利用DTW计算两短音频的相似度
	:param audio_template: 模板音频文件
	:param audio: 待比较音频文件
	:return: DTW距离, float
	"""
	adt_data = parselmouth.Sound(audio_template)
	adt_data.scale_peak(0.99)
	adt_mfcc_obj = adt_data.to_mfcc(number_of_coefficients=12, window_length=0.015, time_step=0.005,
	                                firstFilterFreqency=100.0, distance_between_filters=100.0)
	adt_mfcc = adt_mfcc_obj.to_array().T
	sd_obj = parselmouth.Sound(audio)
	text_grid_vuv = call(sd_obj, "To TextGrid (silences)", 100, 0.0, -25.0, 0.1, 0.1, 'U', 'V')
	vuv_info = call(text_grid_vuv, "List", 'no', 6, 'no', 'no')
	segments_v, segments_u = duration_from_vuvInfo(vuv_info)
	if len(segments_v):  # 头尾静音段不计算在内
		ad_data = call(sd_obj, 'Extract part', segments_v[0][0], segments_v[-1][-1], 'rectangular', 1.0, 'no')
	else:
		ad_data = sd_obj
	ad_data.scale_peak(0.99)
	ad_mfcc_obj = ad_data.to_mfcc(number_of_coefficients=12, window_length=0.015, time_step=0.005,
	                              firstFilterFreqency=100.0, distance_between_filters=100.0)
	ad_mfcc = ad_mfcc_obj.to_array().T
	# praat方法
	# dtw_obj = call([adt_mfcc_obj, ad_mfcc_obj], 'To DTW', 1.0, 0.0, 0.0, 0.0, 0.056, 'no', 'no', 'no restriction')
	# dis = call(dtw_obj, 'Get minimum distance')
	# import fastdtw
	# import dtw
	# from scipy.spatial import distance
	# fastdtw方法，速度第二
	# dis, path = fastdtw.fastdtw(adt_mfcc, ad_mfcc, dist=distance.euclidean)
	# dtw加速方法，速度最慢
	# dis, _, _, path = dtw.accelerated_dtw(adt_mfcc, ad_mfcc, dist=distance.euclidean)
	# dtaidistance加速方法，最快
	dis = dtw_ndim.distance_fast(adt_mfcc, ad_mfcc, use_pruning=True)
	return dis


if __name__ == "__main__":
	import time
	start_time = time.clock()
	sid = os.path.basename(DATA_PATH_EXAMPLE).split('_')[-1]
	audio_file = os.path.join(DATA_PATH_EXAMPLE, f"session_1/09_SRepeating/{sid}_srepeating_1.wav")
	audio_t = os.path.join(parent_path, 'SRepeating_template/01_soothing.wav')
	dtw_dis = speech_similarity(audio_t, audio_file)
	print(dtw_dis)
	print(f"---------- Time Used: {time.clock() - start_time}s ----------")
