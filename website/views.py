# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os #for reading the uploaded file
import pandas as pd #for reading the uploaded file

import random #for sampling the randomly instead of using df.head()



from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

#below package is used for storing the uploaded file into db as a reference 
from django.core.files.storage import FileSystemStorage 
# Create your views here.
#upload_file_url = None



def home(request):
    
    template = loader.get_template('polls/home.html')
    #return HttpResponse("What's up")
    
    if request.method == "POST" and request.FILES['file']:
        
        file_dict = request.FILES['file'] #contains entire line of html code of input in dict
        
        fs = FileSystemStorage()
        
        global file_dict_name
        
        file_dict_name = file_dict.name
        
        my_file = fs.save(file_dict.name, file_dict)
        
        upload_file_url = fs.url(my_file)
        
        #return render(request, 'polls/index.html', {'my_file': my_file} )
        
        return render(request, 'polls/home.html', {'upload_file_url': upload_file_url} )
    
    return render(request, 'polls/home.html')
    

def main(request):
    
    di = os.listdir("D:/python34/task2/website/media")[0]
    #above line of code gives you the list of filenames from that directory
    #since mentioned [0] at the end, it gives you the first file from that dirextory

    df = pd.read_csv(os.path.join("D:/python34/task2/website/media",di))
    
        
    dimen = df.shape
        
    sample_values = random.sample(range(0, dimen[0]),15)
        
    df_sample = pd.DataFrame([df.ix[i,: ] for i in sample_values])
        
    column = [i for i in df_sample.columns]
        
    json_type = df_sample.to_json
        
    context_data_sample = {
            
            'data': json_type,
            'columns': column
            
            }
        
    return render(request, 'polls/home.html', context_data_sample)
    
    """elif request.GET.get('summary'):#for summary of the data
        
        df_summary = df.describe()
        
        column_summary = pd.DataFrame([i for i in df_summary.columns])
        
        data_summary = pd.DataFrame([df_summary[i,:] for i in df_summary])
        
        json_summary = data_summary.to_json
                
        context_summary = {
                
                'data': json_summary,
                'columns': column_summary
                
                }
        
        return render(request, 'polls/home.html', context_summary)
    
    return render(request, 'polls/home.html')

    
    elif request.GET.get('NA_values'):
        
    elif request.GET.get('Moving Averages'):
        # Moving Averages
        
    elif request.Get.get('Exponential Smoothing '):
        # Exponential Smoothing
    
    elif request.Get.get('Double Exponential Smoothing')
        #Double Exponential Smoothing
    
    elif request.Get.get('Triple Exponential Smoothing '):
        #Triple Exponential Smoothing
        
    elif request.Get.get('Holts Linear Exponential Smoothing '):
        #Holts Exponential Smoothing
    """
    


    












