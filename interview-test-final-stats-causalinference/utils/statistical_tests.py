# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 23:55:45 2019

@author: Ying-Fang.Kao
"""

from scipy.stats import f, ttest_ind, levene, ttest_1samp


def one_sample_t_test(sample, popmean = 0):
    
    stats, p_value = ttest_1samp(sample, popmean)
    
    return p_value
    

def two_sample_t_test(sample1, sample2, equal_var = True):
    '''To test whether two samples have the same mean'''
    stats, p_value = ttest_ind(sample1, sample2, equal_var = equal_var)
    
    return p_value


def two_sample_variance_test(sample1, sample2):
    '''To test whether two samples have the same variance'''
    
    stats, p_value = levene(sample1, sample2)
    return p_value