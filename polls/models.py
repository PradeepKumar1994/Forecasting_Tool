# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import pandas as pd
from django.db import models

# Create your models here.
class tab(models.Model):
    
    Date = models.DateField()
    
    Sales = models.CharField(max_length=200)
    
    #last_name = models.CharField(max_length=200)
    #user = models.ForeignKey('auth.User')

    
