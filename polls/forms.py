# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 12:40:27 2018

@author: pradeep kumar
"""

from django import forms


class UploadFileForm(forms.Form):
    
    Upload_File = forms.FileField()


class IntForm(forms.Form):
    
    frequency = forms.IntegerField()


class AlphaForm(forms.Form):
    
    alpha = forms.FloatField()



class DoubleForm(forms.Form):
    
    alpha = forms.FloatField()
    beta  = forms.FloatField()

class TripleForm(forms.Form):
    
    alpha = forms.FloatField()
    beta  = forms.FloatField()
    gamma = forms.FloatField()

class p_q(forms.Form):
    
    p = forms.IntegerField()
    q = forms.IntegerField()

class p_d_q(forms.Form):
    
    p = forms.IntegerField()
    d = forms.IntegerField()
    q = forms.IntegerField()

class p_d_q_s(forms.Form):
    
    p = forms.IntegerField()
    d = forms.IntegerField()
    q = forms.IntegerField()
    P = forms.IntegerField()
    D = forms.IntegerField()
    Q = forms.IntegerField()
    s = forms.IntegerField()


class LinearForm(forms.Form):

    choice_field = forms.ChoiceField(choices=[])

    def __init__(self, *args, **kwargs):
        
        choices = kwargs.pop('choices', [])
        
        super(LinearForm, self).__init__(*args, **kwargs)
        
        self.fields['choice_field'].choices = choices



   
    
class myclass(forms.Form):
    
    all_values = {'p' : forms.IntegerField(),
    'd' : forms.IntegerField(),
    'q' : forms.IntegerField(),
    'P' : forms.IntegerField(),
    'D' : forms.IntegerField(),
    'Q' : forms.IntegerField(),
    's' : forms.IntegerField(),
    }

    
    
    
    
