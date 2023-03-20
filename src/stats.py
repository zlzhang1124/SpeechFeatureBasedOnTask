#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/2/28 17:01 
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : stats.py
# @Software : Python3.6; PyCharm; Windows10 / Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M / 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090
# @Version  : V1.0 - ZL.Z：2022/2/28 - 2022/3/1, 2022/3/4 - 2022/3/5
# 		      First version.
# @License  : None
# @Brief    : 针对重复性数据，计算重测信度、因子分析等统计学指标
import pandas as pd

from src.config import *
import pingouin as pg
from factor_analyzer import FactorAnalyzer, ConfirmatoryFactorAnalyzer, ModelSpecificationParser, \
    calculate_kmo, calculate_bartlett_sphericity
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import warnings
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
warnings.filterwarnings(action='ignore', category=UserWarning, module='factor_analyzer')


def test_retest_icc(feat_file: str, icc_save_file: str):
    """
    test-retest reliability重测信度ICC指标
    :param feat_file: 特征文件
    :param icc_save_file: icc保存文件
    :return: feat_icc所有特征对应的ICC
    """
    feat_pd = pd.read_csv(feat_file)
    feat_icc_all = pd.DataFrame()
    for feat_name in feat_pd.iloc[:, 6:].columns:  # 遍历每一个特征
        # 一次对每一个特征求ICC：targets为id号（同一个id有两次测验），raters为session测试时间点，ratings为特征值
        icc_all = pg.intraclass_corr(data=feat_pd, targets='id', raters='session',
                                     ratings=feat_name, nan_policy='omit')  # 若包含缺失值，则删除该样本再进行ICC
        # 仅保留ICC(3,k):  average measurement, absolute agreement, 2-way mixed effects model
        icc_3k = icc_all[icc_all['Type'] == 'ICC3k']
        icc_3k.insert(0, 'Features', feat_name)
        feat_icc_all = pd.concat([feat_icc_all, icc_3k])
    feat_icc = feat_icc_all.loc[:, ['Features', 'ICC', 'CI95%', 'F', 'df1', 'df2', 'pval']].reset_index(drop=True)
    feat_icc.to_csv(icc_save_file, index=False)
    return feat_icc


def test_retest_icc_gender(feat_file: str, icc_save_file: str):
    """
    test-retest reliability重测信度ICC指标：分性别
    :param feat_file: 特征文件
    :param icc_save_file: icc保存文件
    :return: feat_icc_gender分女男所有特征对应的ICC
    """
    feat_pd = pd.read_csv(feat_file)
    feat_icc_l = []
    for gender in [0, 1]:  # 0女，1男
        feat_icc_all = pd.DataFrame()
        for feat_name in feat_pd.iloc[:, 6:].columns:  # 遍历每一个特征
            # 一次对每一个特征求ICC：targets为id号（同一个id有两次测验），raters为session测试时间点，ratings为特征值
            icc_all = pg.intraclass_corr(data=feat_pd[feat_pd['gender'] == gender],
                                         targets='id', raters='session', ratings=feat_name, nan_policy='omit')
            # 仅保留ICC(3,k):  average measurement, absolute agreement, 2-way mixed effects model
            icc_3k = icc_all[icc_all['Type'] == 'ICC3k']
            icc_3k.insert(0, 'Features', feat_name)
            feat_icc_all = pd.concat([feat_icc_all, icc_3k])
        feat_icc = feat_icc_all.loc[:, ['Features', 'ICC', 'CI95%', 'F', 'df1', 'df2', 'pval']].reset_index(drop=True)
        feat_icc_l.append(feat_icc)
    feat_icc_gender = pd.merge(feat_icc_l[0], feat_icc_l[1], on='Features', suffixes=('-F', '-M'))
    feat_icc_gender.to_csv(icc_save_file, index=False)
    return feat_icc_gender


def exploratory_factor_analysis(feat_file: str = '', excluded_feat=None):
    """
    Exploratory Factor Analysis探索性因子分析（在session2上进行）
    :param feat_file: 特征文件
    :param excluded_feat: 要排除的特征名称列表
    :return: 因子载荷矩阵（旋转后）/KMO检验值/Bartlett检验p值/因子数/共同度矩阵/唯一性矩阵/初始特征值/方差百分比/累计方差
    """
    if excluded_feat is None:
        excluded_feat = []
    feat_pd = pd.read_csv(feat_file)
    _feat_sess1 = feat_pd[feat_pd['session'] == 2].iloc[:, 6:]
    _feat_sess1.drop(columns=excluded_feat, inplace=True)
    feat_sess1 = StandardScaler().fit_transform(_feat_sess1)
    kmo_per_variable, kmo_total = calculate_kmo(feat_sess1)  # 样本充分性检验Kaiser-Meyer-Olkin
    chi_square_value, p_value = calculate_bartlett_sphericity(feat_sess1)  # 巴特利特球形度检验
    _efa = FactorAnalyzer(n_factors=min(feat_sess1.shape[0], feat_sess1.shape[-1]) - 1,
                          rotation='varimax', method='principal')
    _efa.fit(feat_sess1)
    _orig_ev, _common_ev = _efa.get_eigenvalues()  # 初始特征值和公因子特征值，前者对应SPSS中的总方差解释中初始特征值部分
    n_factors = sum(_orig_ev > 1.0)  # 提取初始特征值大于1，作为因子数目
    efa = FactorAnalyzer(n_factors=n_factors, rotation='varimax', method='principal',
                         rotation_kwargs={'max_iter': 1000, 'tol': 1e-10})
    efa.fit(feat_sess1)
    common_factor = efa.get_communalities()  # 共同度矩阵，即每个特征的公共因子，对应SPSS中的公因子方差中的提取部分
    specific_factor = efa.get_uniquenesses()  # 唯一性矩阵，即每个特征的特殊因子
    load_mat = efa.loadings_  # 因子载荷矩阵（旋转后），即SPSS中旋转后的成分矩阵
    orig_ev, common_ev = efa.get_eigenvalues()  # 初始特征值和公因子特征值，前者对应SPSS中的总方差解释中初始特征值部分
    sum_var, prop_var, cum_var = efa.get_factor_variance()  # 因子总计方差、方差百分比、累计方差（旋转载荷），对应SPSS中的总方差解释中旋转载荷平方和部分
    plt.title('Scree Plot', fontdict={'family': font_family, 'size': 16})
    plt.plot(range(1, feat_sess1.shape[1] + 1), orig_ev, marker='.', markersize=10)
    plt.axhline(1.0, c='r', ls='--')
    plt.xlabel('Factors', fontdict={'family': font_family, 'size': 14})
    plt.ylabel('Eigenvalue', fontdict={'family': font_family, 'size': 14})
    plt.xticks(range(1, feat_sess1.shape[1] + 1), fontproperties=font_family, size=10)
    plt.grid()
    plt.show()
    load_mat_pd = pd.DataFrame(load_mat, index=_feat_sess1.columns, columns=[f'FA{i}' for i in range(1, n_factors+1)])
    print('\n\n\n-------------------Exploratory Factor Analysis Result-------------------\n')
    print('\n旋转后的成分矩阵（因子载荷）：\n', load_mat_pd)
    print(f'\nKMO：{kmo_total}；Bartlett-Sphericity：{p_value}；因子数：{n_factors}；\n\n累计方差（旋转载荷平方和）：{cum_var}')
    return load_mat_pd, kmo_total, p_value, n_factors, common_factor, specific_factor, orig_ev, prop_var, cum_var


def confirmatory_factor_analysis(feat_file: str = '', efa_loadings=None, model_dict=None,
                                 excluded_feat=None, use_r_lavaan=False, cfa_res_f: str = ''):
    """
    Confirmatory Factor Analysis验证性因子分析（在session1上进行）
    :param feat_file: 特征文件
    :param efa_loadings: 经过探索性因子分析得到的因子载荷，以从中获取model_dict载荷细节的字典，必须为pd.DataFrame格式
    :param model_dict: 载荷细节的字典，键为公因子名，值为对应的变量名列表（由探索性因子分析的因子载荷而来），当传入efa_loadings，此项无效
    :param excluded_feat: 要排除的特征名称列表
    :param use_r_lavaan: 是否使用R中的lavaan方法计算CFA，若为True则factor_analyzer中的参数efa_loadings和model_dict无效
    :param cfa_res_f: CFA所有结果保存文件，仅使用R中的lavaan方法时有效
    :return: factor_analyzer方法：因子载荷/对数似然/Akaike信息准则/贝叶斯信息准则/因子协方差矩阵/
                                模型隐含的协方差矩阵/因子载荷的标准误/误差方差的标准误
            R中的lavaan方法：无返回值，仅保存至本地txt文件
    """
    if excluded_feat is None:
        excluded_feat = []
    feat_pd = pd.read_csv(feat_file)
    feat_sess2 = feat_pd[feat_pd['session'] == 1].iloc[:, 6:]
    feat_sess2.drop(columns=excluded_feat, inplace=True)
    if efa_loadings is not None:  # 当设置了efa_loadings，则以efa_loadings获取model_dict
        if type(efa_loadings) is not pd.DataFrame:
            raise ValueError('仅接受pd.DataFrame类型的efa_loadings输入')
        _model_dict = {}
        for i_feat in efa_loadings.index:
            _fa = efa_loadings.columns[np.argmax(np.abs(efa_loadings.loc[i_feat]))]
            if _fa in _model_dict.keys():
                _model_dict[_fa].append(i_feat)
            else:
                _model_dict[_fa] = [i_feat]
        model_dict = {k: _model_dict[k] for k in sorted(_model_dict)}
    print('\n\n\n-------------------Confirmatory Factor Analysis Result-------------------\n')
    if use_r_lavaan:
        utils = importr('utils')
        utils.chooseCRANmirror(ind=1)
        names_to_install = [x for x in ('lavaan', ) if not robjects.packages.isinstalled(x)]
        if len(names_to_install) > 0:
            utils.install_packages(StrVector(names_to_install))  # 若没有lavaan包，则安装
        importr('lavaan')
        raw_feat_sess2_colnames = feat_sess2.columns
        feat_sess2.columns = ['x' + str(i) for i in range(16)]  # 转成lavaan需求格式（变量名不能有空格、数字开头等）
        feat_columns_map = pd.DataFrame([feat_sess2.columns], columns=raw_feat_sess2_colnames)  # 保存名字映射
        model_fa = ''
        for i_fa in model_dict.keys():  # lavaan模型输入格式
            model_fa += i_fa + ' =~ ' + ' + '.join([feat_columns_map[i].values[0] for i in model_dict[i_fa]]) + '\n'
        pandas2ri.activate()  # 激活pandas转成R中格式
        robjects.globalenv['feat_sess2'] = feat_sess2  # R环境增加变量
        r_script = f"""
            model <- '{model_fa}'
            fit <- cfa(model, data = feat_sess2, std.lv = TRUE)
            sink(r"({cfa_res_f})")
            summary(fit, fit.measures = TRUE, standardized = TRUE)
            sink()"""  # 定义待执行的R脚本
        robjects.r(r_script)  # 执行R脚本
        with open(cfa_res_f, mode='r', encoding='utf-8') as f:
            cfa_res = f.read()
            for i_feat in raw_feat_sess2_colnames:
                cfa_res = cfa_res.replace(feat_columns_map[i_feat].values[0] + '   ', i_feat)
        cfa_out = f'\nmodel_dict: \n{model_dict}\nmodel: \n{model_fa}\nfeat_columns_map: ' \
                  f'\n{feat_columns_map.to_string(index=False)}\n\nCFA: \n{cfa_res}'
        with open(cfa_res_f, mode='w', encoding='utf-8') as f:
            f.write(cfa_out)
        print(f'{cfa_out}')
    else:
        model_spec = ModelSpecificationParser.parse_model_specification_from_dict(feat_sess2, model_dict)  # 指定CFA模型规范
        cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
        cfa.fit(feat_sess2)
        load_mat = cfa.loadings_  # 因子载荷
        log_lh = cfa.log_likelihood_  # 优化路径的对数似然
        aic = cfa.aic_  # Akaike信息准则
        bic = cfa.bic_  # 贝叶斯信息准则
        varcovs = cfa.factor_varcovs_  # 因子协方差矩阵
        model_implied_cov = cfa.get_model_implied_cov()  # 模型隐含的协方差矩阵(潜变量协方差矩阵)
        loadings_se, error_vars_se = cfa.get_standard_errors()  # 因子载荷的标准误/误差方差的标准误
        load_mat_pd = pd.DataFrame(load_mat, index=feat_sess2.columns,
                                   columns=[f'FA{i}' for i in range(1, len(model_dict)+1)])
        print('因子载荷：\n', load_mat_pd)
        print(f'对数似然: {log_lh}\nAIC: {aic}\nBIC: {bic}\n因子协方差矩阵: {varcovs.shape, varcovs}'
              f'\n因子载荷的标准误: {loadings_se.shape, loadings_se}\n误差方差的标准误: {error_vars_se.shape, error_vars_se}')
        return load_mat_pd, log_lh, aic, bic, varcovs, model_implied_cov, loadings_se, error_vars_se


if __name__ == "__main__":
    res_path = os.path.join(parent_path, r"results")
    feat_f = os.path.join(res_path, 'features.csv')
    _icc_all = test_retest_icc(feat_f, os.path.join(res_path, 'icc_all.csv'))
    _icc_gender = test_retest_icc_gender(feat_f, os.path.join(res_path, 'icc_gender.csv'))
    print(_icc_gender)
    # exc_feat = ['01_SP-APQ', '01_SP-PPQ', '01_SP-DUV(%)', '01_SP-F0 SD(st)', '10_PD-Intensity SD(dB)',
    #             '10_PD-Energy Mean(Pa^2·s)', '11_VDD-Intensity SD(dB)', '11_VDD-Energy Mean(Pa^2·s)',
    #             '12_SI-Intensity SD(dB)', '12_SI-Energy Mean(Pa^2·s)', ]
    icc_all = pd.read_csv(os.path.join(res_path, 'icc_all.csv'))
    exc_feat = icc_all[icc_all['ICC'] <= 0.8]['Features'].tolist()  # 仅针对ICC>0.8的拥有好的重测信度的特征
    loadings = exploratory_factor_analysis(feat_f, exc_feat)[0]
    # 探索性因子分析后，由因子载荷设置载荷细节的字典，键为公因子名，值为对应的变量名列表
    # 方法一（使用factor_analyzer或R中的lavaan方法）：直接传入ml_dict
    # ml_dict = {'FA1': ['01_SP-Formant1(Hz)', '01_SP-Formant2(Hz)', '10_PD-F0 SD(st)', '11_VDD-F0 SD(st)'],
    #            'FA2': ['02_DDK-DDK regularity(ms)', '02_DDK-DDK duration(ms)', '12_SI-Word Sentence Ratio'],
    #            'FA3': ['11_VDD-RST(-/s)', '11_VDD-Voiced Rate(1/s)', '11_VDD-MFCC2', '12_SI-RST(-/s)'],
    #            'FA4': ['01_SP-F0(Hz)', '01_SP-CPP(dB)'],
    #            'FA5': ['01_SP-MPT(s)', '11_VDD-Lexical Density', '12_SI-Function Word Ratio']}
    # confirmatory_factor_analysis(feat_f, model_dict=ml_dict, excluded_feat=exc_feat)
    # 方法二（使用factor_analyzer方法）：传入因子载荷得到ml_dict
    # confirmatory_factor_analysis(feat_f, efa_loadings=loadings, excluded_feat=exc_feat)
    # 方法三（使用R中的lavaan方法）：传入因子载荷得到ml_dict
    confirmatory_factor_analysis(feat_f, efa_loadings=loadings, excluded_feat=exc_feat, use_r_lavaan=True,
                                 cfa_res_f=os.path.join(res_path, 'cfa_res.txt'))


