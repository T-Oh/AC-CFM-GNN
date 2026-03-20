# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:38:12 2023

@author: tobia
"""
import os
from normalization.normalize_DP import normalize_DP
from normalization.normalize_GTSF import normalize_GTSF 


def normalize(cfg_processing, trainset, testset):
    print('normalize')
    print(cfg_processing.normalized_dir)
    if not os.path.exists(cfg_processing.normalized_dir):
        os.makedirs(cfg_processing.normalized_dir)

    if cfg_processing.data_type in ['AC', 'Zhu', 'Zhu_mat73', 'ANGF_Vcf', 'Zhu_nobustype']:
        normalize_DP(cfg_processing, trainset, testset)
    elif cfg_processing.data_type in ['LSTM']:
        normalize_GTSF(cfg_processing, trainset, testset)

    

    