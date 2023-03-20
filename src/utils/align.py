#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/3/16 16:19
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: 1. https://github.com/jaekookang/p2fa_py3
#             2. https://github.com/chenchenzi/P2FA_Mandarin_py3
# @FileName : align.py
# @Software : Python3.6; PyCharm; Windows10 / Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M / 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090
# @Version  : V1.0 - ZL.Z：2022/3/16 - 2022/3/17
# 		      First version.
# @License  : None
# @Brief    : 中文音频文本强制对齐，利用Penn Forced Aligner (P2FA)

import subprocess
import wave
import re
import codecs
import shutil
import textgrid
from src.config import *


MODEL_DIR = P2FA_MODEL
try:
    HTK_DIR = os.path.relpath(HTK_PATH, os.getcwd())
except ValueError:
    HTK_DIR = HTK_PATH
LOG_LIKELIHOOD_REGEX = r'.+==\s+\[\d+ frames\]\s+(-?\d+.\d+)'


def prep_mlf(trsfile: str, temp_dir: str):
    """
    根据label词汇文本生成待处理的mlf文件
    :param trsfile: 转录的词汇文本文件
    :param temp_dir: 生成待处理的mlf文件的临时路径
    :return: unks词典中未定义的词汇
    """
    with codecs.open(temp_dir + '/dict_final', 'r', 'utf-8') as f:
        lines = f.readlines()
    _dict = []
    for line in lines:
        _dict.append(line.split()[0])
    with codecs.open(temp_dir + '/puncs', 'r', 'utf-8') as f:
        lines = f.readlines()
    puncs = []
    for line in lines:
        puncs.append(line.strip())
    with codecs.open(trsfile, 'r', 'utf-8') as f:
        lines = f.readlines()
    with codecs.open(temp_dir + f'/{os.path.basename(trsfile)}.mlf', 'w', 'utf-8') as fw:
        fw.write('#!MLF!#\n')
        fw.write(f'"*/{os.path.basename(trsfile)}"\n')
        fw.write('sp\n')
        i = 0
        unks = set()
        while i < len(lines):
            txt = lines[i].replace('\n', '')
            txt = txt.replace('{breath}', 'br').replace('{noise}', 'ns')
            txt = txt.replace('{laugh}', 'lg').replace('{laughter}', 'lg')
            txt = txt.replace('{cough}', 'cg').replace('{lipsmack}', 'ls')
            for pun in puncs:
                txt = txt.replace(pun,  '')
            for wrd in txt.split():
                if wrd in _dict:
                    fw.write(wrd + '\n')
                    fw.write('sp\n')
                else:
                    unks.add(wrd)
            i += 1
        fw.write('.\n')
    return unks


def gen_res(infile1, infile2, outfile):
    """
    根据HVite生成的aligned.mlf获取正确编码的汉语mlf结果
    :param infile1: HVite生成的aligned.mlf文件
    :param infile2: 生成待处理的mlf文件
    :param outfile: 正确编码的汉语mlf文件
    :return: None
    """
    with codecs.open(infile1, 'r', 'utf-8') as f:
        lines = f.readlines()
    with codecs.open(infile2, 'r', 'utf-8') as f:
        lines2 = f.readlines()
    words = []
    for line in lines2[2:-1]:
        if line.strip() != 'sp':
            words.append(line.strip())
    words.reverse()
    with codecs.open(outfile, 'w', 'utf-8') as fw:
        fw.write(lines[0])
        fw.write(lines[1])
        for line in lines[2:-1]:
            if (line.split()[-1].strip() == 'sp') or (len(line.split()) != 5):
                fw.write(line)
            else:
                fw.write(line.split()[0] + ' ' + line.split()[1] + ' ' + line.split()
                         [2] + ' ' + line.split()[3] + ' ' + words.pop() + '\n')
        fw.write(lines[-1])


def read_mlf(mlffile, sr):
    """
    根据正确编码的汉语mlf文件获取词汇对齐分割列表
    :param mlffile: 正确编码的汉语mlf文件
    :param sr: 音频采样率
    :return: 正确编码的汉语mlf文件词汇对齐后的分割列表
    """
    # This reads a MLFalignment output  file with phone and word
    # alignments and returns a list of words, each word is a list containing
    # the word label followed by the phones, each phone is a tuple
    # (phone, start_time, end_time) with times in seconds.
    with codecs.open(mlffile, 'r', 'utf-8') as f:
        lines = [l.rstrip() for l in f.readlines()]
    if len(lines) < 3:
        raise ValueError("Alignment did not complete succesfully.")
    j = 2
    ret = []
    while lines[j] != '.':
        if len(lines[j].split()) == 5:
            # Is this the start of a word; do we have a word label?
            # Make a new word list in ret and put the word label at the beginning
            wrd = lines[j].split()[4]
            ret.append([wrd])
        # Append this phone to the latest word (sub-)list
        ph = lines[j].split()[2]
        if sr == 11025:
            st = (float(lines[j].split()[0]) / 10000000.0 + 0.0125) * (11000.0 / 11025.0)
            en = (float(lines[j].split()[1]) / 10000000.0 + 0.0125) * (11000.0 / 11025.0)
        else:
            st = float(lines[j].split()[0]) / 10000000.0 + 0.0125
            en = float(lines[j].split()[1]) / 10000000.0 + 0.0125
        if st < en:
            ret[-1].append([ph, st, en])
        j += 1
    return ret


def write_text_grid(outfile, word_alignments):
    """
    将正确编码的汉语mlf文件词汇对齐后的分割列表转成Praat TextGrids文件
    :param outfile: 转成的Praat TextGrids文件路径
    :param word_alignments: 正确编码的汉语mlf文件词汇对齐后的分割列表
    :return: None
    """
    # make the list of just phone alignments
    phons = []
    for wrd in word_alignments:
        phons.extend(wrd[1:])  # skip the word label
    # make the list of just word alignments
    # we're getting elements of the form:
    #   ["word label", ["phone1", start, end], ["phone2", start, end], ...]
    wrds = []
    for wrd in word_alignments:
        # If no phones make up this word, then it was an optional word
        # like a pause that wasn't actually realized.
        if len(wrd) == 1:
            continue
        wrds.append([wrd[0], wrd[1][1], wrd[-1][2]])  # word label, first phone start time, last phone end time
    with open(outfile, 'w', encoding='utf-8') as fw:
        fw.write('File type = "ooTextFile short"\n')
        fw.write('"TextGrid"\n')
        fw.write('\n')
        fw.write(str(phons[0][1]) + '\n')
        fw.write(str(phons[-1][-1]) + '\n')
        fw.write('<exists>\n')
        fw.write('2\n')
        # write the phone interval tier
        fw.write('"IntervalTier"\n')
        fw.write('"phone"\n')
        fw.write(str(phons[0][1]) + '\n')
        fw.write(str(phons[-1][-1]) + '\n')
        fw.write(str(len(phons)) + '\n')
        for k in range(len(phons)):
            fw.write(str(phons[k][1]) + '\n')
            fw.write(str(phons[k][2]) + '\n')
            fw.write('"' + phons[k][0] + '"' + '\n')
        # write the word interval tier
        fw.write('"IntervalTier"\n')
        fw.write('"word"\n')
        fw.write(str(phons[0][1]) + '\n')
        fw.write(str(phons[-1][-1]) + '\n')
        fw.write(str(len(wrds)) + '\n')
        for k in range(len(wrds) - 1):
            fw.write(str(wrds[k][1]) + '\n')
            fw.write(str(wrds[k+1][1]) + '\n')
            fw.write('"' + wrds[k][0] + '"' + '\n')
        fw.write(str(wrds[-1][1]) + '\n')
        fw.write(str(phons[-1][2]) + '\n')
        fw.write('"' + wrds[-1][0] + '"' + '\n')


def get_av_log_likelihood_per_frame(file_path):
    with codecs.open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    score = re.match(LOG_LIKELIHOOD_REGEX, lines[-1]).groups()[0]
    return float(score)


def align(wavfile, trsfile, outfile=None, temp_dir=None, miss_w_dir=None,
          dict_alone_f=None, dict_add_f=None, puncs_f=None):
    """
    汉语的音频文本强制对齐
    :param wavfile: 待对齐的音频文件
    :param trsfile: 待对齐的转录文本文件
    :param outfile: 输出的Praat TextGrids格式对齐文件
    :param temp_dir: 保存中间步骤的临时文件夹
    :param miss_w_dir: 字典中未存在的丢失词文件保存路径
    :param dict_alone_f: 单独使用的替换字典文件
    :param dict_add_f: 额外添加的字典文件
    :param puncs_f: 排除的标点文件
    :return: align_res_tg最终的对齐结果 textgrid.textgrid.TextGrid格式：首元素为phone音素级别的对齐，另一元素为word词级别的对齐，
    每个级别对齐中的每个元素为一个word对齐结果，Interval(minTime, maxTime, mark)格式，可以用align_res_tg[i][j].minTime/maxTime/mark访问
    """
    # create working directory
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    # find sampling rate and prepare wavefile
    wavtemp = os.path.join(temp_dir, os.path.basename(wavfile))
    f = wave.open(wavfile, 'r')
    sr = f.getframerate()
    f.close()
    if sr not in [8000, 16000]:
        subprocess.run(f"ffmpeg -loglevel quiet -y -i {wavfile} -ar 16000 {wavtemp}", shell=True)
        sr = 16000
    else:
        shutil.copy(wavfile, temp_dir)
    # prepare dictionary
    if dict_alone_f:
        with codecs.open(dict_alone_f, 'r', 'utf-8') as f:
            lines = f.readlines()
        lines = lines + ['sp sp\n']
    else:
        with codecs.open(MODEL_DIR + '/dict', 'r', 'utf-8') as f:
            lines = f.readlines()
        if dict_add_f:
            with codecs.open(dict_add_f, 'r', 'utf-8')as f:
                lines2 = f.readlines()
            lines = lines + lines2
    with codecs.open(os.path.join(temp_dir, 'dict_final'), 'w', 'utf-8') as f:
        for line in lines:
            f.write(line)
    # generate the plp file using a given configuration file for HCopy
    subprocess.run(f"{os.path.join(HTK_DIR, 'HCopy')} -C {os.path.join(MODEL_DIR, str(sr), 'config')} "
                   f"{wavtemp} {wavtemp.replace('.wav', '.plp')}", shell=True)
    if puncs_f:
        shutil.copy(puncs_f, temp_dir)
    else:
        shutil.copy(MODEL_DIR + '/puncs', temp_dir)
    unks = prep_mlf(trsfile, temp_dir)
    if miss_w_dir is None:
        miss_w_dir = temp_dir
    with open(miss_w_dir + '/miss_wrd_align.txt', 'a', encoding='utf-8') as f:
        for unk in unks:
            f.write(unk + '\n')
    # run Verterbi decoding and alignment
    mpfile = os.path.join(MODEL_DIR, 'monophones')
    macros_dir = os.path.join(MODEL_DIR, str(sr), 'macros')
    hmmdefs_dir = os.path.join(MODEL_DIR, str(sr), 'hmmdefs')
    output_mlf = os.path.join(temp_dir, 'aligned.mlf')
    results_mlf = os.path.join(temp_dir, 'aligned.results')  #
    subprocess.run(f"{os.path.join(HTK_DIR, 'HVite')} -T 1 -a -m -t 10000.0 10000.0 100000.0 "
                   f"-I {os.path.join(temp_dir, os.path.basename(trsfile))}.mlf "
                   f"-H {macros_dir} -H {hmmdefs_dir} "
                   f"-i {output_mlf} {os.path.join(temp_dir, 'dict_final')} "
                   f"{mpfile} {wavtemp.replace('.wav', '.plp')} > {results_mlf}", shell=True)
    mlffile = wavtemp.replace('.wav', '.mlf')
    gen_res(output_mlf, temp_dir + f'/{os.path.basename(trsfile)}.mlf', mlffile)
    # output the alignment as a Praat TextGrid
    write_text_grid(outfile, read_mlf(mlffile, sr))
    align_res_tg = textgrid.TextGrid.fromFile(outfile, name='align_phone-word')
    # av_score_per_frame = get_av_log_likelihood_per_frame(results_mlf)
    # clean directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    return align_res_tg
