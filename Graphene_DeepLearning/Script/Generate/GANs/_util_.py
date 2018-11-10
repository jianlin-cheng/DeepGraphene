#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 00:32:30 2018

@author: Herman Wu
"""
import os
import Master_GANs
from pandas.core.frame import DataFrame

Local_dir = os.path.dirname(__file__)
Base_dir=(Local_dir.split('Script'))[0]
csv_dir=Base_dir+'GANs_result/result.csv'

epoch=[1000]

rela_error=[]
abs_error=[]
per_error=[]
for i in epoch:
    [rela_e,abs_e,per_e]=Master_GANs.main(Epc=i)
    rela_error.append(rela_e)
    abs_error.append(abs_e)
    per_error.append(per_e)

data={'epoch':epoch,'Relative error':rela_error,'Absolute error':abs_error,'Percent error':per_error}
csv_data=DataFrame(data)

csv_data.to_csv(csv_dir)