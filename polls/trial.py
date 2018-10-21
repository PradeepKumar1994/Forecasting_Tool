# -*- coding: utf-8 -*-
"""
Created on Tue Nov 07 19:02:11 2017

@author: VAANI-P
"""
import os
import pandas as pd



di = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'media')

#print di

a =  os.listdir(di)[0]
#print a


c = os.path.join(di, a)

B = pd.read_csv(c)

#print c

