# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 11:03:53 2018

@author: pradeep kumar
"""
import django_tables2 as tables


class d_tables(tables.Table):
    
    class Meta:
        
        model = tab
        
