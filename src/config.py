#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2021/8/13 15:17 
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : config.py
# @Software : Python3.6; PyCharm; Windows10
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M
# @Version  : V1.0 - ZL.Z：2021/8/13 - 2021/8/13
# 		      First version.
# @License  : None
# @Brief    : 配置文件

import os
import platform
import pandas as pd
import matplotlib

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
os.environ['OUTDATED_IGNORE'] = '1'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

_current_path = os.path.dirname(os.path.realpath(__file__))  # 获取当前文件的路径
parent_path = os.path.dirname(_current_path)
grandpa_path = os.path.dirname(parent_path)
if platform.system() == 'Windows':
	font_family = 'Times New Roman'
	# font_family = 'Arial'
	DATA_PATH = r"F:\Graduate\NeurocognitiveAssessment\认知与声音\言语特征可重复性\data\preprocessed_data"
	TRANS_PATH = r'F:\Graduate\NeurocognitiveAssessment\认知与声音\言语特征可重复性\data\transcription'
	CUR_CODE_DATA_SAVE_PATH = r"F:\Graduate\NeurocognitiveAssessment\认知与声音\言语特征可重复性\analysis\AcousticFeatureBasedOnTask\data"
	# DICTIONARY_PATH = r'D:\pretrained_models\MFA\dictionary\mandarin_pinyin.dict'
	# ACOUSTIC_MODEL_PATH = r'D:\pretrained_models\MFA\acoustic\mandarin.zip'
	HTK_PATH = os.path.join(parent_path, "tools/Windows/HTK-3.4/bin.win32")
	PRETRAINED_MODEL_fastText = r'D:\pretrained_models\fastText\wiki.zh.bin'
	MODEL_PATH_HanLP = r'D:\pretrained_models\hanlp'
else:
	font_family = 'DejaVu Sans'
	DATA_PATH = r"/home/zlzhang/data/言语特征可重复性/data/preprocessed_data"
	TRANS_PATH = r'/home/zlzhang/data/言语特征可重复性/data/transcription'
	CUR_CODE_DATA_SAVE_PATH = r"/home/zlzhang/data/言语特征可重复性/analysis/AcousticFeatureBasedOnTask/data"
	# DICTIONARY_PATH = r'/home/zlzhang/pretrained_models/MFA/dictionary/mandarin_pinyin.dict'
	# ACOUSTIC_MODEL_PATH = r'/home/zlzhang/pretrained_models/MFA/acoustic/mandarin.zip'
	HTK_PATH = os.path.join(parent_path, "tools/Linux/HTK-3.4/bin")
	# HTK_PATH = ''
	PRETRAINED_MODEL_fastText = r'/home/zlzhang/pretrained_models/fastText/wiki.zh.bin'
	MODEL_PATH_HanLP = r'/home/zlzhang/pretrained_models/hanlp'
os.environ['HANLP_HOME'] = MODEL_PATH_HanLP
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
matplotlib.rcParams["font.family"] = font_family
subj = 'HC/male/202201070908_20220107001'
# subj = 'HC/female/202201081557_20220108009'
DATA_PATH_EXAMPLE = os.path.join(CUR_CODE_DATA_SAVE_PATH, subj)
TRANS_PATH_EXAMPLE = os.path.join(TRANS_PATH, 'ASR', subj)
P2FA_MODEL = os.path.join(parent_path, "tools/P2FA_model/Mandarin")
TEMP_PATH = os.path.join(parent_path, "results/temp")
