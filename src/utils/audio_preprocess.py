#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2021/8/13 17:11
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : audio_preprocess.py
# @Software : Python3.6; PyCharm; Windows10
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M
# @Version  : V1.1: 2022/2/26
#             适应可重复性测试
#             V1.0: 2021/8/13
#             First version.
# @License  : None
# @Brief    : 语音预处理

import subprocess
import shutil
import parselmouth
from parselmouth.praat import call
import warnings
import datetime
from pathos.pools import ProcessPool as Pool
from src.config import *


def audio_preprocess(audio_file: str, output_folder: str, start=0.0, denoise=False):
    """
    音频预处理：包括格式转换、嘀声删除以及降噪
    :param audio_file: 待处理音频文件
    :param output_folder: 输出文件夹
    :param start: 删除前start s的音频（数据收集问题，导致将嘀声收集到原始文件，此时要删除）
    :param denoise: 是否进行降噪，由于降噪对于不同音频效果有差异，默认否
    :return: None
    """
    print(audio_file + "音频数据处理中...")
    if not os.path.exists(audio_file):
        raise FileExistsError(audio_file + "输入文件不存在！")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    audio_name = os.path.basename(audio_file)
    output_audio = os.path.join(output_folder, audio_name)
    temp_audio = os.path.join(output_folder, "temp_" + audio_name)
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
    print("----------{} STEP1: 音频格式转换----------".format(audio_name))
    # 调用ffmpeg，将任意格式音频文件从start到最后转换为.wav文件，pcm有符号16bit,1：单通道,16kHz，不显示打印信息
    subprocess.run("ffmpeg -loglevel quiet -ss %f -y -i %s -acodec pcm_s16le -ac 1 -ar 16000 %s" %
                   (start, audio_file, temp_audio), shell=True)
    print("----------{} STEP2: 降噪----------".format(audio_name))
    sound_obj = parselmouth.Sound(temp_audio)
    sound_obj.subtract_mean()  # 消除直流分量
    if denoise:
        sound_denoised_obj = call(sound_obj, "Reduce noise", 0.0, 0.0, 0.025, 80.0,
                                  10000.0, 40.0, -20.0, "spectral-subtraction")
        sound_obj_final = sound_denoised_obj
    else:
        sound_obj_final = sound_obj
    with warnings.catch_warnings(record=True) as w:
        sound_obj_final.save(output_audio, parselmouth.SoundFileFormat.WAV)
        if len(w):
            # print(f'{w[0].category}: {w[0].message}')
            sound_obj_final.scale_peak(0.99)
            sound_obj_final.save(output_audio, parselmouth.SoundFileFormat.WAV)
    if os.path.exists(temp_audio):
        os.remove(temp_audio)


def run_audio_preprocess_parallel(original_path: str, preprocessed_path: str, n_jobs=None):
    """
    并行运行音频预处理
    :param original_path: 原始数据文件路径
    :param preprocessed_path: 预处理保存数据文件路径
    :param n_jobs: 并行运行CPU核数，默认为None，取os.cpu_count()全部核数,-1/正整数/None类型
    :return: None
    """
    assert (n_jobs is None) or (type(n_jobs) is int and n_jobs > 0) or (n_jobs == -1), 'n_jobs仅接受-1/正整数/None类型输入'
    if n_jobs == -1:
        n_jobs = None
    for each_file in os.listdir(original_path):
        if each_file == "HC":
            preprocessed_p = os.path.join(preprocessed_path, each_file)
            if not os.path.exists(preprocessed_p):
                os.makedirs(preprocessed_p)
            data_path = os.path.join(original_path, each_file)
            for root, dirs, files in os.walk(data_path):
                def parallel_process(name):
                    if name.endswith('.csv'):  # 将csv文件复制到目标文件夹下
                        csv_file = os.path.join(root, name)
                        output_path = root.replace(os.path.abspath(original_path), os.path.abspath(preprocessed_path))
                        if not os.path.exists(output_path):
                            os.makedirs(output_path)
                        shutil.copy(csv_file, os.path.join(output_path, name))
                    if name.endswith('.wav'):  # 遍历处理.wav文件
                        wav_file = os.path.join(root, name)
                        output_path = root.replace(os.path.abspath(original_path), os.path.abspath(preprocessed_path))
                        audio_preprocess(wav_file, output_path)

                # 使用设定数量的CPU核数（这里有闭包，不可pickle，因此不能用multiprocessing中的Pool，这里用pathos）
                with Pool(n_jobs) as pool:
                    pool.map(parallel_process, files)


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print(f"---------- Start Time ({os.path.basename(__file__)}): {start_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
    original_data = DATA_PATH  # 原始数据文件夹
    # if os.path.exists(CUR_CODE_DATA_SAVE_PATH):
    #     shutil.rmtree(CUR_CODE_DATA_SAVE_PATH)
    run_audio_preprocess_parallel(original_data, CUR_CODE_DATA_SAVE_PATH, n_jobs=-1)

    end_time = datetime.datetime.now()
    print(f"---------- End Time ({os.path.basename(__file__)}): {end_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
    print(f"---------- Time Used ({os.path.basename(__file__)}): {end_time - start_time} ----------")
    with open(os.path.join(parent_path, r"./results/finished.txt"), "w") as ff:
        ff.write(f"------------------ Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
        ff.write(f"------------------ Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
        ff.write(f"------------------ Time Used {end_time - start_time} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")

