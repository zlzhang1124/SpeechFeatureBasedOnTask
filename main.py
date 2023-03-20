#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2021/5/17 18:50
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: 1. https://github.com/jcvasquezc/DisVoice
#             2. https://github.com/jcvasquezc/NeuroSpeech
#             3. https://github.com/MLSpeech/Dr.VOT
#             4. Rusz, J., et al. (2018). "Smartphone Allows Capture of Speech Abnormalities Associated
#             With High Risk of Developing Parkinson’s Disease." IEEE Transactions on Neural Systems and
#             Rehabilitation Engineering 26(8): 1495-1507.
#             5. Novotny, M., Rusz, J., Cmejla, R., & Ruzicka, E. (2014). Automatic Evaluation of Articulatory
#             Disorders in Parkinson’s Disease. IEEE/ACM Transactions on Audio, Speech, and Language
#             Processing, 22(9), 1366-1378. doi:10.1109/taslp.2014.2329734
#             6. Hlavnicka, J., Cmejla, R., Tykalova, T., Sonka, K., Ruzicka, E., & Rusz, J. (2017).
#             Automated analysis of connected speech reveals early biomarkers of Parkinson's disease
#             in patients with rapid eye movement sleep behaviour disorder. Sci Rep, 7(1), 12.
#             doi:10.1038/s41598-017-00047-5
# @FileName : main.py
# @Software : Python3.6; PyCharm; Windows10
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M
# @Version  : V1.8.0 - ZL.Z：2022/4/24， 2022/5/5
# 		      1. 增加双任务范式对应的特征：DDK-WM和PD-连续减法任务
#             2. 对多次测试的任务（SP/DDK/DDKWM），输入支持文件夹，当为文件夹输入时，计算的特征为该文件夹下所有对应音频的特征均值
#             3. 添加use_参数，并将parallel参数合并至n_jobs=1情形.
#             V1.6.0 - ZL.Z：2022/4/7
# 		      1. 修改SI任务，添加19个特征，包括：
# 		      1). 基于声学的特征：Hesitation Ratio;
# 		      2). 句法特征（Morpho-syntactic & Syntactic）：Noun Phrase Rate/Verb Phrase Rate/Adj Phrase Rate/
# 		      Adv Phrase Rate/Prep Phrase Rate/Parse Tree Height/Yngve Depth Max/Yngve Depth Mean/
# 		      Yngve Depth Total/Frazier Depth Max/Frazier Depth Mean/Frazier Depth Total/
# 		      Dependency Distance Total/Dependency Distance Mean;
# 		      3). 语篇和语用（Discourse & Pragmatic）特征：Discourse Total/Discourse Rate/Information Units/Efficiency.
# 		      2. 增加task参数实现指定任务的特征提取；
# 		      3. 增加parallel参数实现并行与非并行的选择.
#             V1.5.0: 2022/3/13 - 2022/3/25, 2022/3/28 - 2022/3/30
#             1. 根据分析结果，将一些分类贡献度低、ICC低(>0.6，并不是所有)的特征做标记隐藏处理（方法前加_标记）；
#             2. 增加04_Stroop/05_VFT/06_Naming/07_SS任务的相关特征；
#             3. 修改文件架构
#             V1.4.0: 2022/2/27, 2022/3/5
#             1. 清除临床量表相关；
#             2. 适配新的采集软件并添加若干新特征，包括一些文本特征.
#             V1.3.1: 2021/10/22
#             1. 添加新增的数据;
#             2. 特证名字精简与修正。
#             V1.3.0: 2021/10/15
#             1. 恢复V1.1.0, 仅保留mean特征,排除未成年人;
#             2. 去掉CVP中的VSA和FCR特征以及SZR中的S/Z difference;
#             3. frenchay分级3/4级合并为重度3级.
#             V1.2.0: 2021/9/29
#             仅保留mean特征;去掉CVP特征;增加MPT额外特征;frenchay分级3/4级合并为重度3级.
#             V1.1.0: 2021/8/13 - 2021/8/20
#             仅构音任务的声学特征提取.
#             V1.0: 2021/5/17 - 2021/5/25
#             First version.
# @License  : None
# @Brief    : 针对快速言语认知评估采集到的的数据进行声学和语言特征提取

import glob
import datetime
from pathos.pools import ProcessPool as Pool
from src.utils.util import *
from src.task.SP import SPFeatures
from src.task.DDK import DDKFeatures
from src.task.SZR import SZRFeatures
from src.task.Stroop import StroopFeatures
from src.task.VFT import VFTFeatures
from src.task.Naming import NamingFeatures
from src.task.SS import SSFeatures
from src.task.SReading import SReadingFeatures
from src.task.SRepeating import speech_similarity
from src.task.PD import PDFeatures
from src.task.VDD import VDDFeatures
from src.task.SI import SIFeatures
from src.task.DDKWM import DDKWMFeatures
from src.task.PDSS import PDSSFeatures
from src.utils.audio_preprocess import run_audio_preprocess_parallel
from src.stats import *


class FeatureExtract:
    """并行运行特征提取程序"""
    def __init__(self, audio_data_dir: str = '', trans_data_dir: str = '', res_file: str = '',
                 temp_dir: str = '', miss_w_dir: str = '', task: list = None, use_=True):
        """
        初始化
        :param audio_data_dir: 音频数据文件路径
        :param trans_data_dir: 对应的转录文本数据文件路径，仅参数task包含需要文本的任务可用
        :param res_file: 结果保存文件
        :param temp_dir: 中间结果临时保存路径，仅参数task包含05_VFT任务可用
        :param miss_w_dir: 所用的字典中未存在的丢失词文件保存路径（包括对齐用的字典和动物词汇字典），仅参数task包含05_VFT任务可用
        :param task: 待提取的任务，list类型，仅提取该列表中的任务对应的特征，默认为None，即提取全部任务的特征
        :param use_: 是否提取包含分类贡献度低、ICC低的（方法有_标记）特征，默认是
        :return: None
        """
        self.trans_data_dir = trans_data_dir
        tasks = ['01_SP', '02_DDK', '03_SZR', '04_Stroop', '05_VFT', '06_Naming', '07_SS', '08_SReading',
                 '09_SRepeating', '10_PD', '11_VDD', '12_SI', '13_DDKDT', '14_PDSS']
        if task is None or task == []:
            task = tasks
        assert set(task).issubset(set(tasks)), f'仅接受以下任务集中所列：{tasks}'
        self.task = task
        self.use_ = use_
        # 特征文件csv的表头，需严格按照各个任务的特征顺序排序
        self.head0 = ["id", "session", "name", "edu", "age", "gender"]
        self.head1 = ["01_SP-Jitter(%)", "01_SP-Shimmer(%)", "01_SP-HNR(dB)", "01_SP-F0(Hz)", "01_SP-F0 SD(st)",
                      "01_SP-PPQ", "01_SP-APQ", "01_SP-DUV(%)", "01_SP-Formant1(Hz)", "01_SP-Formant2(Hz)", "01_SP-MPT(s)",
                      "01_SP-CPP(dB)"]
        self.head2 = ["02_DDK-DDK rate(syll/s)", "02_DDK-DDK regularity(ms)", "02_DDK-DDK duration(ms)", "02_DDK-VOT(ms)",
                      "02_DDK-pause rate(1/s)", "02_DDK-pause regularity(ms)", "02_DDK-pause duration(ms)"]
        self.head3 = ["03_SZR-SZR"]
        self.head4 = ["04_Stroop-Time(s)", "04_Stroop-Correct Count", "04_Stroop-SIE Time(s)", "04_Stroop-SIE Count"]
        self.head5 = ["05_PFT-Word Count", "05_PFT-Repetitions", "05_PFT-Intrusions",
                      "05_PFT-WC Bin1", "05_PFT-WC Bin2", "05_PFT-WC Bin3", "05_PFT-TD Bin1", "05_PFT-TD Bin2",
                      "05_PFT-TD Bin3", "05_PFT-SD Bin1", "05_PFT-SD Bin2", "05_PFT-SD Bin3", "05_PFT-LD Bin1",
                      "05_PFT-LD Bin2", "05_PFT-LD Bin3", "05_PFT-Cluster Size", "05_PFT-Cluster Switches",
                      "05_PFT-Word Frequency(%)", "05_SFT-Word Count", "05_SFT-Repetitions", "05_SFT-Intrusions",
                      "05_SFT-WC Bin1", "05_SFT-WC Bin2", "05_SFT-WC Bin3", "05_SFT-TD Bin1", "05_SFT-TD Bin2",
                      "05_SFT-TD Bin3", "05_SFT-SD Bin1", "05_SFT-SD Bin2", "05_SFT-SD Bin3", "05_SFT-LD Bin1",
                      "05_SFT-LD Bin2", "05_SFT-LD Bin3", "05_SFT-Cluster Size", "05_SFT-Cluster Switches",
                      "05_SFT-Word Frequency(%)"]
        self.head6 = ['06_Naming-Time Mean(s)', '06_Naming-Time SD(s)', '06_Naming-Accuracy']
        self.head7 = ["07_SS-Time(s)", "07_SS-Accuracy30", "07_SS-Correct Count"]
        self.head8 = ["08_SReading-F0 SD(st)", "08_SReading-Intensity SD(dB)",
                      "08_SReading-DPI(ms)", "08_SReading-RST(-/s)", "08_SReading-EST", "08_SReading-Voiced Rate(1/s)",
                      "08_SReading-Hesitation Ratio", "08_SReading-Energy Mean(Pa^2·s)", "08_SReading-MFCC2",
                      "08_SReading-Word Rate(-/s)"]
        self.head9 = ["09_SRepeating-similarity"]
        self.head10 = ["10_PD-F0 SD(st)", "10_PD-Intensity SD(dB)", "10_PD-DPI(ms)", "10_PD-RST(-/s)",
                       "10_PD-EST", "10_PD-Voiced Rate(1/s)", "10_PD-Hesitation Ratio",
                       "10_PD-Energy Mean(Pa^2·s)", "10_PD-MFCC2", "10_PD-Word Num",
                       "10_PD-Word Rate(-/s)", "10_PD-Function Word Ratio", "10_PD-Lexical Density",
                       "10_PD-MATTR", "10_PD-Sentence Num", "10_PD-MLU", "10_PD-Noun Phrase Rate",
                       "10_PD-Verb Phrase Rate", "10_PD-Adj Phrase Rate", "10_PD-Adv Phrase Rate",
                       "10_PD-Prep Phrase Rate", "10_PD-Parse Tree Height", "10_PD-Yngve Depth Max",
                       "10_PD-Yngve Depth Mean", "10_PD-Yngve Depth Total", "10_PD-Frazier Depth Max",
                       "10_PD-Frazier Depth Mean", "10_PD-Frazier Depth Total", "10_PD-Dependency Distance Total",
                       "10_PD-Dependency Distance Mean", "10_PD-Discourse Total", "10_PD-Discourse Rate",
                       "10_PD-Information Units", "10_PD-Efficiency"]
        self.head11 = ["11_VDD-F0 SD(st)", "11_VDD-Intensity SD(dB)", "11_VDD-DPI(ms)", "11_VDD-RST(-/s)", "11_VDD-EST",
                       "11_VDD-Voiced Rate(1/s)", "11_VDD-Hesitation Ratio", "11_VDD-Energy Mean(Pa^2·s)", "11_VDD-MFCC2",
                       "11_VDD-Word Num", "11_VDD-Word Rate(-/s)", "11_VDD-Function Word Ratio", "11_VDD-Lexical Density",
                       "11_VDD-MATTR", "11_VDD-Sentence Num", "11_VDD-MLU", "11_VDD-Noun Phrase Rate",
                       "11_VDD-Verb Phrase Rate", "11_VDD-Adj Phrase Rate", "11_VDD-Adv Phrase Rate",
                       "11_VDD-Prep Phrase Rate", "11_VDD-Parse Tree Height", "11_VDD-Yngve Depth Max",
                       "11_VDD-Yngve Depth Mean", "11_VDD-Yngve Depth Total", "11_VDD-Frazier Depth Max",
                       "11_VDD-Frazier Depth Mean", "11_VDD-Frazier Depth Total", "11_VDD-Dependency Distance Total",
                       "11_VDD-Dependency Distance Mean", "11_VDD-Discourse Total", "11_VDD-Discourse Rate",
                       "11_VDD-Information Units", "11_VDD-Efficiency"]
        self.head12 = ["12_SI-F0 SD(st)", "12_SI-Intensity SD(dB)", "12_SI-DPI(ms)", "12_SI-RST(-/s)", "12_SI-EST",
                       "12_SI-Voiced Rate(1/s)", "12_SI-Hesitation Ratio", "12_SI-Energy Mean(Pa^2·s)", "12_SI-MFCC2",
                       "12_SI-Word Num", "12_SI-Word Rate(-/s)", "12_SI-Function Word Ratio", "12_SI-Lexical Density",
                       "12_SI-MATTR", "12_SI-Sentence Num", "12_SI-MLU", "12_SI-Noun Phrase Rate",
                       "12_SI-Verb Phrase Rate", "12_SI-Adj Phrase Rate", "12_SI-Adv Phrase Rate",
                       "12_SI-Prep Phrase Rate", "12_SI-Parse Tree Height", "12_SI-Yngve Depth Max",
                       "12_SI-Yngve Depth Mean", "12_SI-Yngve Depth Total", "12_SI-Frazier Depth Max",
                       "12_SI-Frazier Depth Mean", "12_SI-Frazier Depth Total", "12_SI-Dependency Distance Total",
                       "12_SI-Dependency Distance Mean", "12_SI-Discourse Total", "12_SI-Discourse Rate",
                       "12_SI-Information Units", "12_SI-Efficiency"]
        self.head13 = ["13_DDKWM-DTC DDK rate(%)", "13_DDKWM-DTC DDK regularity(%)", "13_DDKWM-DTC DDK duration(%)", 
                       "13_DDKWM-DTC VOT(%)", "13_DDKWM-DTC pause rate(%)", "13_DDKWM-DTC pause regularity(%)", 
                       "13_DDKWM-DTC pause duration(%)", "13_DDKWM-WM Correct Count"]
        self.head14 = ["14_PDSS-DTC F0 SD(%)", "14_PDSS-DTC Intensity SD(%)", "14_PDSS-DTC DPI(%)", "14_PDSS-DTC RST(%)",
                       "14_PDSS-DTC EST(%)", "14_PDSS-DTC Voiced Rate(%)", "14_PDSS-DTC Hesitation Ratio(%)",
                       "14_PDSS-DTC Energy Mean(%)", "14_PDSS-MFCC2", "14_PDSS-DTC Word Num(%)",
                       "14_PDSS-DTC Word Rate(%)", "14_PDSS-DTC Function Word Ratio(%)", "14_PDSS-DTC Lexical Density(%)",
                       "14_PDSS-DTC MATTR(%)", "14_PDSS-DTC Sentence Num(%)", "14_PDSS-DTC MLU(%)", "14_PDSS-DTC Noun Phrase Rate(%)",
                       "14_PDSS-DTC Verb Phrase Rate(%)", "14_PDSS-DTC Adj Phrase Rate(%)", "14_PDSS-DTC Adv Phrase Rate(%)",
                       "14_PDSS-DTC Prep Phrase Rate(%)", "14_PDSS-DTC Parse Tree Height(%)", "14_PDSS-DTC Yngve Depth Max(%)",
                       "14_PDSS-DTC Yngve Depth Mean(%)", "14_PDSS-DTC Yngve Depth Total(%)", "14_PDSS-DTC Frazier Depth Max(%)",
                       "14_PDSS-DTC Frazier Depth Mean(%)", "14_PDSS-DTC Frazier Depth Total(%)", "14_PDSS-DTC Dependency Distance Total(%)",
                       "14_PDSS-DTC Dependency Distance Mean(%)", "14_PDSS-DTC Discourse Total(%)", "14_PDSS-DTC Discourse Rate(%)",
                       "14_PDSS-DTC Information Units(%)", "14_PDSS-DTC Efficiency(%)",
                       "14_PDSS-Accuracy12", "14_PDSS-Correct Count"]
        self.head = self.head0
        for i_task in self.task:
            self.head += eval(f'self.head{tasks.index(i_task) + 1}')
        self.res_file = res_file
        self.temp_dir = temp_dir
        self.miss_w_dir = miss_w_dir
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        if os.path.exists(self.miss_w_dir):
            shutil.rmtree(self.miss_w_dir, ignore_errors=True)
        self.subject_dir_list = []  # 每个被试的音频主文件路径组成的列表
        for i_each_file in os.listdir(audio_data_dir):
            data_path_group = os.path.join(audio_data_dir, i_each_file)
            for j_each_file in os.listdir(data_path_group):
                data_path_gender = os.path.join(data_path_group, j_each_file)
                for k_each_file in os.listdir(data_path_gender):
                    data_path_sess = os.path.join(data_path_gender, k_each_file)
                    for l_each_file in os.listdir(data_path_sess):
                        self.subject_dir_list.append(os.path.join(data_path_sess, l_each_file))

    def extract(self, subject_dir: str):
        """
        提取全部特征，对于同一个任务存在多个测试音频文件情况，最后的特征求均值处理
        :param subject_dir: 被试主文件夹路径
        :return: 所有特征集, pd.DataFrame类型
        """
        print("---------- Processing %d / %d: %s ----------" %
              (self.subject_dir_list.index(subject_dir) + 1, len(self.subject_dir_list), subject_dir))
        subject_id = os.path.normpath(subject_dir).split(os.sep)[-2].split("_")[1]
        gender = os.path.normpath(subject_dir).split(os.sep)[-3]
        rec_f = subject_dir + "/" + subject_id + ".csv"
        csv_data = read_csv(rec_f)
        name = csv_data[0][0].split("：")[1]
        age = int(csv_data[0][1].split("：")[1])
        edu = float(csv_data[0][5].split("：")[1])
        sess = int(csv_data[0][4].split("：")[1])
        feat_id = pd.DataFrame.from_dict({"id": [subject_id], "session": [sess], "name": [name], "edu": [edu], "age": [age],
                                          "gender": [{"male": 1, "female": 0}[gender]]}, orient='index').transpose()
        sp_feat, ddk_feat, szr_feat, stroop_feat, vft_feat, naming_feat, ss_feat, reading_feat, \
        repeating_feat, pd_feat, vdd_feat, si_feat, ddkwm_feat, pdss_feat = \
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), \
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), \
            pd.DataFrame(), pd.DataFrame()
        for each_file in os.listdir(subject_dir):
            if each_file == "01_SP" and each_file in self.task:  # 提取SP任务的声学特征
                vowel_audios = os.path.join(subject_dir, each_file)
                if len(os.listdir(vowel_audios)):
                    sp_feat = SPFeatures(vowel_audios).get_all_feat(prefix=f'{each_file}-', use_=self.use_)
            elif each_file == "02_DDK" and each_file in self.task:  # 提取DDK任务的声学特征
                ddk_audios = os.path.join(subject_dir, each_file)
                if len(os.listdir(ddk_audios)):
                    ddk_feat = DDKFeatures(ddk_audios).get_all_feat(prefix=f'{each_file}-', use_=self.use_)
            elif each_file == "03_SZR" and each_file in self.task:  # 提取SZR任务的声学特征
                szr_dir_s = os.path.join(subject_dir, each_file + "/" + subject_id + "_s")
                szr_dir_z = os.path.join(subject_dir, each_file + "/" + subject_id + "_z")
                audio_s_dict, audio_z_dict = {}, {}
                try:
                    if os.listdir(szr_dir_s) and os.listdir(szr_dir_z):
                        for szr_audio_s in os.listdir(szr_dir_s):
                            audio_s_dict[os.path.join(szr_dir_s, szr_audio_s)] = \
                                parselmouth.Sound(os.path.join(szr_dir_s, szr_audio_s)).get_total_duration()
                        for szr_audio_z in os.listdir(szr_dir_z):
                            audio_z_dict[os.path.join(szr_dir_z, szr_audio_z)] = \
                                parselmouth.Sound(os.path.join(szr_dir_z, szr_audio_z)).get_total_duration()
                    try:
                        audio_s = list(audio_s_dict.keys())[list(audio_s_dict.values()).index(max(list(audio_s_dict.values())))]
                        audio_z = list(audio_z_dict.keys())[list(audio_z_dict.values()).index(max(list(audio_z_dict.values())))]
                        szr_feat = pd.DataFrame({f"{each_file}-SZR": [SZRFeatures(audio_s, audio_z).sz_ratio()]})
                    except ValueError:
                        szr_feat = pd.DataFrame({f"{each_file}-SZR": [np.nan]})
                except FileNotFoundError:
                    szr_feat = pd.DataFrame({f"{each_file}-SZR": [np.nan]})
            elif each_file == "04_Stroop" and each_file in self.task:  # 提取Stroop任务的特征
                stroop_audio_dir = os.path.join(subject_dir, each_file)
                if len(os.listdir(stroop_audio_dir)):
                    try:
                        stroop_trans_dir = glob.glob(
                            os.path.join(self.trans_data_dir,
                                         f'**/{os.path.normpath(subject_dir).split(os.sep)[-2]}'
                                         f'/session_{sess}/{each_file}'), recursive=True)[0]
                    except IndexError:
                        raise FileNotFoundError(f'{subject_id}: session_{sess}/{each_file} 文件夹不存在，但对应音频文件夹存在')
                    stroop_feat = StroopFeatures(stroop_audio_dir, stroop_trans_dir).get_all_feat(prefix=f'{each_file}-', use_=self.use_)
            elif each_file == "05_VFT" and each_file in self.task:  # 提取VFT任务的特征
                pft_feat, sft_feat = pd.DataFrame(), pd.DataFrame()
                pft_audio = os.path.join(subject_dir, each_file, "phonetic", subject_id + "_phonetic.wav")
                sft_audio = os.path.join(subject_dir, each_file, "semantic", subject_id + "_semantic.wav")
                if os.path.exists(pft_audio):
                    try:
                        pft_trans = glob.glob(os.path.join(self.trans_data_dir,
                                                           f'**/session_{sess}/**/{subject_id}_phonetic.txt'),
                                              recursive=True)[0]
                    except IndexError:
                        raise FileNotFoundError(f'session_{sess}/**/{subject_id}_phonetic.txt 不存在，但对应音频存在')
                    pft_feat = VFTFeatures(pft_audio, pft_trans, self.temp_dir,
                                           self.miss_w_dir, True).get_all_feat(prefix='05_PFT-', use_=self.use_)
                if os.path.exists(sft_audio):
                    try:
                        sft_trans = glob.glob(os.path.join(self.trans_data_dir,
                                                           f'**/session_{sess}/**/{subject_id}_semantic.txt'),
                                              recursive=True)[0]
                    except IndexError:
                        raise FileNotFoundError(f'session_{sess}/**/{subject_id}_semantic.txt 不存在，但对应音频存在')
                    sft_feat = VFTFeatures(sft_audio, sft_trans, self.temp_dir,
                                           self.miss_w_dir, False).get_all_feat(prefix='05_SFT-', use_=self.use_)
                vft_feat = pd.concat([pft_feat, sft_feat], axis=1)
            elif each_file == "06_Naming" and each_file in self.task:  # 提取Naming任务的特征
                naming_audio_dir = os.path.join(subject_dir, each_file)
                if len(os.listdir(naming_audio_dir)):
                    try:
                        naming_trans_dir = glob.glob(
                            os.path.join(self.trans_data_dir,
                                         f'**/{os.path.normpath(subject_dir).split(os.sep)[-2]}'
                                         f'/session_{sess}/{each_file}'), recursive=True)[0]
                    except IndexError:
                        raise FileNotFoundError(f'{subject_id}: session_{sess}/{each_file} 文件夹不存在，但对应音频文件夹存在')
                    naming_feat = NamingFeatures(naming_audio_dir, naming_trans_dir).get_all_feat(
                        prefix=f'{each_file}-', use_=self.use_)
            elif each_file == "07_SS" and each_file in self.task:  # 提取SS任务的特征
                ss_audio = os.path.join(subject_dir, each_file, subject_id + "_ss.wav")
                if os.path.exists(ss_audio):
                    try:
                        ss_trans = glob.glob(os.path.join(self.trans_data_dir,
                                                          f'**/session_{sess}/**/{subject_id}_ss.txt'),
                                             recursive=True)[0]
                    except IndexError:
                        raise FileNotFoundError(f'session_{sess}/**/{subject_id}_ss.txt 不存在，但对应音频存在')
                    ss_feat = SSFeatures(ss_audio, ss_trans).get_all_feat(prefix=f'{each_file}-', use_=self.use_)
            elif each_file == "08_SReading" and each_file in self.task:  # 提取SReading任务的特征
                reading_feat_n = pd.DataFrame()
                for num in range(1, 7):
                    reading_audio = os.path.join(subject_dir, each_file, subject_id + f"_sreading_{num}.wav")
                    if os.path.exists(reading_audio):
                        try:
                            reading_trans = glob.glob(os.path.join(self.trans_data_dir,
                                                                   f'**/session_{sess}/**/{subject_id}_sreading_{num}.txt'),
                                                      recursive=True)[0]
                        except IndexError:
                            raise FileNotFoundError(f'session_{sess}/**/{subject_id}_sreading_{num}.txt 不存在，但对应音频存在')
                        reading_feat_n = pd.concat([reading_feat_n, SReadingFeatures(reading_audio, reading_trans).
                                                   get_all_feat(prefix=f'{each_file}-')], ignore_index=True)
                reading_feat = pd.DataFrame(reading_feat_n.mean()).T
            elif each_file == "09_SRepeating" and each_file in self.task:  # 提取SRepeating任务的特征
                audio_t = [os.path.join(current_path, 'SRepeating_template/01_soothing.wav'),
                           os.path.join(current_path, 'SRepeating_template/02_surging.wav'),
                           os.path.join(current_path, 'SRepeating_template/03_cheerful.wav')]
                dtw_dis = []
                for num in range(1, 4):
                    repeating_audio = os.path.join(subject_dir, each_file, subject_id + f"_srepeating_{num}.wav")
                    if os.path.exists(repeating_audio):
                        dtw_dis.append(speech_similarity(audio_t[num-1], repeating_audio))
                dis_mean = np.mean(dtw_dis)
                repeating_feat = pd.DataFrame({f"{each_file}-similarity": [dis_mean]})
            elif each_file == "10_PD" and each_file in self.task:  # 提取PD任务的特征
                pd_audio = os.path.join(subject_dir, each_file, "CookieTheft", subject_id + "_CookieTheft.wav")
                if os.path.exists(pd_audio):
                    try:
                        pd_trans = glob.glob(os.path.join(self.trans_data_dir,
                                                          f'**/session_{sess}/**/{subject_id}_CookieTheft.txt'),
                                             recursive=True)[0]
                    except IndexError:
                        raise FileNotFoundError(f'session_{sess}/**/{subject_id}_CookieTheft.txt 不存在，但对应音频存在')
                    pd_feat_all = PDFeatures(pd_audio, pd_trans).get_all_feat(prefix=f'{each_file}-', use_=self.use_)
                    pd_feat_all.drop(columns=f'{each_file}-TF-IDF', inplace=True)
                    mfcc_mean = np.mean(PDFeatures(pd_audio, pd_trans).mfcc(), axis=1)
                    pd_feat = pd.concat([pd_feat_all.drop(columns=f"{each_file}-MFCC"),
                                         pd.DataFrame({f"{each_file}-MFCC2": [mfcc_mean[1]]})], axis=1)
            elif each_file == "11_VDD" and each_file in self.task:  # 提取VDD任务的特征
                vdd_audio = os.path.join(subject_dir, each_file, subject_id + "_vdd.wav")
                if os.path.exists(vdd_audio):
                    try:
                        vdd_trans = glob.glob(os.path.join(self.trans_data_dir,
                                                           f'**/session_{sess}/**/{subject_id}_vdd.txt'),
                                              recursive=True)[0]
                    except IndexError:
                        raise FileNotFoundError(f'session_{sess}/**/{subject_id}_vdd.txt 不存在，但对应音频存在')
                    vdd_feat_all = VDDFeatures(vdd_audio, vdd_trans).get_all_feat(prefix=f'{each_file}-', use_=self.use_)
                    vdd_feat_all.drop(columns=f'{each_file}-TF-IDF', inplace=True)
                    mfcc_mean = np.mean(VDDFeatures(vdd_audio, vdd_trans).mfcc(), axis=1)
                    vdd_feat = pd.concat([vdd_feat_all.drop(columns=f"{each_file}-MFCC"),
                                          pd.DataFrame({f"{each_file}-MFCC2": [mfcc_mean[1]]})], axis=1)
            elif each_file == "12_SI" and each_file in self.task:  # 提取SI任务的特征
                si_audio = os.path.join(subject_dir, each_file, subject_id + "_si.wav")
                if os.path.exists(si_audio):
                    try:
                        si_trans = glob.glob(os.path.join(self.trans_data_dir,
                                                          f'**/session_{sess}/**/{subject_id}_si.txt'),
                                             recursive=True)[0]
                    except IndexError:
                        raise FileNotFoundError(f'session_{sess}/**/{subject_id}_si.txt 不存在，但对应音频存在')
                    si_feat_all = SIFeatures(si_audio, si_trans).get_all_feat(prefix=f'{each_file}-', use_=self.use_)
                    si_feat_all.drop(columns=f'{each_file}-TF-IDF', inplace=True)
                    mfcc_mean = np.mean(SIFeatures(si_audio, si_trans).mfcc(), axis=1)
                    si_feat = pd.concat([si_feat_all.drop(columns=f"{each_file}-MFCC"),
                                         pd.DataFrame({f"{each_file}-MFCC2": [mfcc_mean[1]]})], axis=1)
            elif each_file == "13_DDKDT" and each_file in self.task:  # 提取DDKWM任务的声学特征
                ddk_audios_dt = os.path.join(subject_dir, each_file)
                ddk_audios_sig = os.path.join(subject_dir, "02_DDK")
                if len(os.listdir(ddk_audios_dt)) and len(os.listdir(ddk_audios_sig)):
                    for i_dt in range(1, len(os.listdir(ddk_audios_dt)) + 1):
                        ddkwm_feat = pd.concat([ddkwm_feat, DDKWMFeatures(ddk_audios_dt, ddk_audios_sig, rec_f, i_dt).
                                                get_all_feat(prefix=f'{each_file}{i_dt}-', use_=self.use_)], axis=1)
            elif each_file == "14_PDSS" and each_file in self.task:  # 提取PDSS任务的特征
                pd_audio_dt = os.path.join(subject_dir, each_file, subject_id + "_pd.wav")
                pd_audio_sig = os.path.join(subject_dir, "10_PD/CookieTheft", subject_id + "_CookieTheft.wav")
                ss_audio_dt = os.path.join(subject_dir, each_file, subject_id + "_ss.wav")
                if os.path.exists(pd_audio_dt) and os.path.exists(pd_audio_sig) and os.path.exists(ss_audio_dt):
                    try:
                        pd_trans_dt = glob.glob(
                            os.path.join(self.trans_data_dir,
                                         f'**/session_{sess}/{each_file}/{subject_id}_pd.txt'),
                            recursive=True)[0]
                    except IndexError:
                        raise FileNotFoundError(f'session_{sess}/{each_file}/{subject_id}_pd.txt 不存在，但对应音频存在')
                    try:
                        pd_trans_sig = glob.glob(
                            os.path.join(self.trans_data_dir,
                                         f'**/session_{sess}/10_PD/{subject_id}_CookieTheft.txt'), recursive=True)[0]
                    except IndexError:
                        raise FileNotFoundError(f'session_{sess}/10_PD/{subject_id}_CookieTheft.txt 不存在，但对应音频存在')
                    try:
                        ss_trans_dt = glob.glob(
                            os.path.join(self.trans_data_dir,
                                         f'**/session_{sess}/{each_file}/{subject_id}_ss.txt'),
                            recursive=True)[0]
                    except IndexError:
                        raise FileNotFoundError(f'session_{sess}/{each_file}/{subject_id}_ss.txt 不存在，但对应音频存在')
                    pdss_feat_all = PDSSFeatures(pd_audio_dt, pd_trans_dt, ss_audio_dt, ss_trans_dt, pd_audio_sig,
                                                 pd_trans_sig).get_all_feat(prefix=f'{each_file}-', use_=self.use_)
                    pdss_feat_all.drop(columns=f'{each_file}-TF-IDF', inplace=True)
                    mfcc_mean = np.mean(PDSSFeatures(pd_audio_dt, pd_trans_dt, ss_audio_dt, ss_trans_dt,
                                                     pd_audio_sig, pd_trans_sig).mfcc(), axis=1)
                    pdss_feat = pd.concat([pdss_feat_all.drop(columns=f"{each_file}-MFCC"),
                                           pd.DataFrame({f"{each_file}-MFCC2": [mfcc_mean[1]]})], axis=1)

        _acoustic_feat = pd.concat([feat_id, sp_feat, ddk_feat, szr_feat, stroop_feat, vft_feat, naming_feat,
                                    ss_feat, reading_feat, repeating_feat, pd_feat, vdd_feat, si_feat,
                                    ddkwm_feat, pdss_feat], axis=1)
        acoustic_feat = pd.concat([pd.DataFrame(columns=self.head), _acoustic_feat])  # 防止有的音频不存在导致对应列数据不存在
        acoustic_feat.fillna(np.nan, inplace=True)
        acoustic_feat.dropna(axis=1, how='all', inplace=True)  # 删除整列都为空的特征
        return acoustic_feat

    def run(self, n_jobs=None):
        """
        并行处理，保存所有特征至csv文件
        :param n_jobs: 并行运行CPU核数，默认为None;若为1非并行，若为-1或None,取os.cpu_count()全部核数,-1/正整数/None类型
        :return: None
        """
        assert (n_jobs is None) or (type(n_jobs) is int and n_jobs > 0) or (
                n_jobs == -1), 'n_jobs仅接受-1/正整数/None类型输入'
        if n_jobs == -1:
            n_jobs = None
        if n_jobs == 1:
            res = []
            for i_subj in self.subject_dir_list:
                res.append(self.extract(i_subj))
        else:
            with Pool(n_jobs) as pool:
                res = pool.map(self.extract, self.subject_dir_list)
        acoustic_feats_all = pd.DataFrame()
        for _res in res:
            acoustic_feats_all = pd.concat([acoustic_feats_all, _res], ignore_index=True)
        acoustic_feats_all.sort_values(by=['id', 'session'], inplace=True)
        acoustic_feats_all = acoustic_feats_all[acoustic_feats_all['age'] >= 18]  # 去掉未成年数据
        acoustic_feats_all.to_csv(self.res_file, encoding="utf-8-sig", index=False)


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print(f"---------- Start Time ({os.path.basename(__file__)}): {start_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
    current_path = os.getcwd()  # 获取当前文件夹
    original_data = DATA_PATH  # 原始数据文件夹
    trans_path = os.path.join(TRANS_PATH, 'ASR')
    res_path = os.path.join(current_path, r"results")
    temp_path = os.path.join(res_path, 'temp')
    miss_wd_path = os.path.join(res_path, 'missing_words')
    feat_f = os.path.join(res_path, 'features.csv')

    audio_preprocess_flag = False  # 是否进行音频预处理(仅需进行一次，之后改为False)
    if audio_preprocess_flag:
        if os.path.exists(CUR_CODE_DATA_SAVE_PATH):
            shutil.rmtree(CUR_CODE_DATA_SAVE_PATH)
        run_audio_preprocess_parallel(original_data, CUR_CODE_DATA_SAVE_PATH, n_jobs=-1)

    # 特征提取
    FeatureExtract(CUR_CODE_DATA_SAVE_PATH, trans_path, feat_f,
                   temp_path, miss_wd_path, task=None, use_=True).run(n_jobs=1)

    # ICC计算
    icc_all = test_retest_icc(feat_f, os.path.join(res_path, 'icc_all.csv'))
    test_retest_icc_gender(feat_f, os.path.join(res_path, 'icc_gender.csv'))

    # EFA和CFA分析
    # exc_feat = icc_all[icc_all['ICC'] < 0.8]['Features'].tolist()  # 仅针对ICC>0.8的拥有好的重测信度的特征
    # loadings = exploratory_factor_analysis(feat_f, exc_feat)[0]
    # # 探索性因子分析后，由因子载荷设置载荷细节的字典，键为公因子名，值为对应的变量名列表
    # # 方法一（使用factor_analyzer或R中的lavaan方法）：直接传入ml_dict
    # # ml_dict = {'FA1': ['01_SP-Formant1(Hz)', '01_SP-Formant2(Hz)', '10_PD-F0 SD(st)', '11_VDD-F0 SD(st)'],
    # #            'FA2': ['02_DDK-DDK regularity(ms)', '02_DDK-DDK duration(ms)', '12_SI-Word Sentence Ratio'],
    # #            'FA3': ['11_VDD-RST(-/s)', '11_VDD-Voiced Rate(1/s)', '11_VDD-MFCC2', '12_SI-RST(-/s)'],
    # #            'FA4': ['01_SP-F0(Hz)', '01_SP-CPP(dB)'],
    # #            'FA5': ['01_SP-MPT(s)', '11_VDD-Lexical Density', '12_SI-Function Word Ratio']}
    # # confirmatory_factor_analysis(feat_f, model_dict=ml_dict, excluded_feat=exc_feat)
    # # 方法二（使用factor_analyzer方法）：传入因子载荷得到ml_dict
    # # confirmatory_factor_analysis(feat_f, efa_loadings=loadings, excluded_feat=exc_feat)
    # # 方法三（使用R中的lavaan方法）：传入因子载荷得到ml_dict
    # confirmatory_factor_analysis(feat_f, efa_loadings=loadings, excluded_feat=exc_feat, use_r_lavaan=True,
    #                              cfa_res_f=os.path.join(res_path, 'cfa_res.txt'))

    end_time = datetime.datetime.now()
    print(f"---------- End Time ({os.path.basename(__file__)}): {end_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
    print(f"---------- Time Used ({os.path.basename(__file__)}): {end_time - start_time} ----------")
    with open(r"./results/finished.txt", "w") as ff:
        ff.write(f"------------------ Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
        ff.write(f"------------------ Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
        ff.write(f"------------------ Time Used {end_time - start_time} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
