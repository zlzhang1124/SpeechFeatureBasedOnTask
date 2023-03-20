#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/3/29 21:45
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : SS.py
# @Software : Python3.6; PyCharm; Windows10
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M
# @Version  : V1.1.0 - ZL.Z：2022/5/19
# 		      使用To TextGrid (silences)方法替换To TextGrid (vuv)，仅分割有声和无声段
#             V1.0 - ZL.Z：2022/3/29 - 2022/3/30
# 		      First version.
# @License  : None
# @Brief    : 提取SS,100连续减3任务的特征: Time(s)/Accuracy30/Correct Count

from src.utils.util import *
from src.config import *


class SSFeatures:
    """获取SS任务的特征"""
    def __init__(self, input_f_audio: str, input_f_trans: str, f0min=75, f0max=600):
        """
        初始化
        :param input_f_audio: 输入.wav音频文件，或是praat所支持的文件格式
        :param input_f_trans: 输入文本转录文件，.txt或.cha类似的文件格式
        :param f0min: 最小追踪pitch,默认75Hz
        :param f0max: 最大追踪pitch,默认600Hz
        """
        self.f_audio = input_f_audio
        self.f_trans = input_f_trans
        self.f0min = f0min
        self.f0max = f0max
        if self.f_trans.endswith('.txt'):
            try:
                with open(self.f_trans, "r", encoding="utf-8") as f:
                    self.text = f.read()
            except UnicodeDecodeError:
                with open(self.f_trans, "r", encoding="gb18030") as f:
                    self.text = f.read()
        self.cor_num_l = self.func_correct_word()  # 对转录文本进行校正后的数字列表

    def func_correct_word(self):
        """
        校正ASR转录的文本
        :return: 校正后的转录文本列表：100-3数字列表
        """
        num_str = ''
        num_l = []
        for i_text in self.text:  # 将文本转为数字列表
            if i_text.isdigit() or (i_text in CN_NUM.keys()) or (i_text in CN_UNIT.keys()):  # 数字
                num_str += i_text
            else:
                if num_str:
                    num_l.append(int(cn2an(num_str)))  # 确保中文数字被转换
                num_str = ''
        num_l_cor = []
        for ind in range(len(num_l)):
            if ind == 0 and num_l[ind] > 100:  # 首元素经常出现197，即100和97连在一起
                num_l_cor.append(100)
                num_l_cor.append(num_l[ind] % 100)
            # 针对第2到倒数第5个元素，经常出现≥4位数字，如3936，即39和36被ASR自动合并成了3936
            elif len(num_l) - ind > 4 and len(str(num_l[ind])) >= 4:
                if int(num_l[ind] / 100) > 1000:
                    num_l_cor.append(int(num_l[ind] / 10000))
                    num_l_cor.append(int(num_l[ind] % 10000 / 100))
                else:
                    num_l_cor.append(int(num_l[ind] / 100))
                num_l_cor.append(num_l[ind] % 100)
            # 针对第2到倒数第5个元素，经常出现3位数字，如402，即42被ASR自动拆分并合并成了402
            elif len(num_l) - ind > 4 and len(str(num_l[ind])) == 3 and str(num_l[ind])[1] == '0':
                num_l_cor.append(int(str(num_l[ind])[0] + str(num_l[ind])[-1]))
            # 针对第2到倒数第5个元素，经常出现连续两个数字，前面一个为某0，后一个为单位数，如40,2，即42被ASR自动拆分成了40和2
            # 此时，前一个数字：结合后一个数字组成校正后的数字
            elif len(num_l) - ind > 4 and len(str(num_l[ind])) == 2 and str(num_l[ind])[-1] == '0':
                try:
                    if num_l[ind + 1] < 10:
                        num_l_cor.append(int(str(num_l[ind])[0] + str(num_l[ind + 1])))
                    else:
                        num_l_cor.append(num_l[ind])
                except IndexError:
                    num_l_cor.append(num_l[ind])
            # 后一个数字：直接删除进行校正（当第2到倒数第5个元素之间出现一位数时，也会直接删除）
            elif len(num_l) - ind > 3 and len(str(num_l[ind])) == 1:
                try:
                    if len(str(num_l[ind - 1])) == 2 and str(num_l[ind - 1])[-1] == '0':
                        pass
                except IndexError:
                    num_l_cor.append(num_l[ind])
            # 针对最后4个元素，经常出现数字连在一起，如10741代表10/7/4/1，963代表9/6/3
            elif len(num_l) - ind <= 4:
                if num_l[ind] >= 30:  # 考虑到100-3最后的可能连在一起的最小数字为30
                    if str(num_l[ind]).startswith('10'):  # 10开头则10单独为一个数字。剩下再分开
                        num_l_cor.append(10)
                        for _i in range(len(str(num_l[ind]).lstrip('10'))):
                            num_l_cor.append(int(str(num_l[ind]).lstrip('10')[_i]))
                    elif 0 < int(num_l[ind] / 100) - num_l[ind] % 100 <= 5:  # 两个两位数连在一起：2219
                        num_l_cor.append(int(num_l[ind] / 100))
                        num_l_cor.append(num_l[ind] % 100)
                    else:
                        for _i in range(len(str(num_l[ind]))):
                            num_l_cor.append(int(str(num_l[ind])[_i]))
                elif len(str(num_l[ind])) == 2 and str(num_l[ind])[0] == '1':  # 10和单位数连在一起：17代表10/7
                    try:
                        if num_l[ind] > num_l[ind - 1]:  # 若该位数字比前一位数字大，则判定为连在一起
                            num_l_cor.append(10)
                            num_l_cor.append(int(str(num_l[ind])[-1]))
                        else:
                            num_l_cor.append(num_l[ind])
                    except IndexError:
                        num_l_cor.append(num_l[ind])
                else:
                    num_l_cor.append(num_l[ind])
            else:
                num_l_cor.append(num_l[ind])
        return num_l_cor

    def total_time_use(self):
        """
        计算总用时
        :return: 总用时，单位s
        """
        sd_obj = parselmouth.Sound(self.f_audio)
        text_grid_vuv = call(sd_obj, "To TextGrid (silences)", 100, 0.0, -25.0, 0.1, 0.1, 'U', 'V')
        vuv_info = call(text_grid_vuv, "List", 'no', 6, 'no', 'no')
        segments_v, segments_u = duration_from_vuvInfo(vuv_info)
        if len(segments_v):  # 为浊音
            time_len = segments_v[-1][-1] - segments_v[0][0]
        else:  # 为清音
            time_len = segments_u[-1][-1] - segments_u[0][0]
        return time_len

    def _total_num_correct(self, start_number=100, difference_value=3):
        """
        计算数字列表的总正确数，默认为100-3
        :param start_number: 开始数字，默认100
        :param difference_value: 差值，默认3
        :return: 总正确数
        """
        correct_num = 0
        number_list_copy = [self.cor_num_l[0]]  # 拷贝原始数字列表，第一位不变
        for number_list_index in range(1, len(self.cor_num_l)):  # 从第二位开始判断相邻元素是否重复
            if self.cor_num_l[number_list_index] != self.cor_num_l[number_list_index - 1]:  # 不重复则加入拷贝列表
                number_list_copy.append(self.cor_num_l[number_list_index])
        for number_list_copy_index in range(len(number_list_copy)):  # 计算相邻数值差
            if number_list_copy_index == 0 and number_list_copy[0] == start_number:  # 当首位为开始数值时
                continue  # 从下一个开始计算
            else:  # 否则依次计算
                if number_list_copy_index == 0:  # 若为首位
                    dif_val = start_number - number_list_copy[0]  # 从首位便开始计算
                else:  # 否则依次计算
                    dif_val = number_list_copy[number_list_copy_index - 1] - \
                              number_list_copy[number_list_copy_index]
            if dif_val == difference_value:  # 计算正确
                correct_num += 1
        return correct_num

    def accuracy(self, start_number=100, difference_value=3, n_num=30):
        """
        计算数字列表的前n个数字的准确率，默认为100-3（准确计算完会产生33个数字，为了标准化且固定范围避免较长计算，这里只取前n个数字）
        :param start_number: 开始数字，默认100
        :param difference_value: 差值，默认3
        :param n_num: 为了标准化且固定范围避免较长计算，这里只取前n个数字，默认前30个
        :return: 前n个数字的准确率
        """
        correct_num = 0
        _n_num = n_num
        number_list_copy = [self.cor_num_l[0]]  # 拷贝原始数字列表，第一位不变
        for number_list_index in range(1, len(self.cor_num_l)):  # 从第二位开始判断相邻元素是否重复
            if self.cor_num_l[number_list_index] != self.cor_num_l[number_list_index - 1]:  # 不重复则加入拷贝列表
                number_list_copy.append(self.cor_num_l[number_list_index])
        for number_list_copy_index in range(len(number_list_copy)):  # 计算相邻数值差
            if number_list_copy_index == 0 and number_list_copy[0] == start_number:  # 当首位为开始数值时
                _n_num += 1
                continue  # 从下一个开始计算
            else:  # 否则依次计算
                if number_list_copy_index == 0:  # 若为首位
                    dif_val = start_number - number_list_copy[0]  # 从首位便开始计算
                else:  # 否则依次计算
                    dif_val = number_list_copy[number_list_copy_index - 1] - \
                              number_list_copy[number_list_copy_index]
            if dif_val == difference_value and number_list_copy_index < _n_num:  # 计算正确，且只计入前30个
                correct_num += 1
        return correct_num / n_num

    def get_all_feat(self, prefix='SS_', use_=False):
        """
        获取当前所有特征
        :param prefix: pd.DataFrame类型特征列名的前缀
        :param use_: 是否提取包含分类贡献度低、ICC低的（方法有_标记）特征，默认否
        :return: 该类的全部特征, pd.DataFrame类型
        """
        time_use = self.total_time_use()
        acc = self.accuracy()
        if use_:
            num_cor = self._total_num_correct()
            feat = {prefix+"Time(s)": [time_use], prefix+"Accuracy30": [acc],
                    prefix+"Correct Count": [num_cor], }
        else:
            feat = {prefix+"Time(s)": [time_use], prefix + "Accuracy30": [acc], }
        return pd.DataFrame(feat)


if __name__ == "__main__":
    sid = os.path.basename(DATA_PATH_EXAMPLE).split('_')[-1]
    audio_file = os.path.join(DATA_PATH_EXAMPLE, f"session_1/07_SS/{sid}_ss.wav")
    trans_file = os.path.join(TRANS_PATH_EXAMPLE, f"session_1/07_SS/{sid}_ss.txt")
    ss_feat = SSFeatures(audio_file, trans_file).get_all_feat()
    print(ss_feat)





