# -*- coding: utf-8 -*-
""" Created on Wed Nov 01 11:30:09 2017
@author: pradeep kumar
"""
#from __future__ import unicode_literals
#For packages used check under PACKAGES in the documentation
import numpy as np, pandas as pd

import scipy as sp
import scipy.stats

import plotly.offline as opy
import plotly.graph_objs as go

from sklearn.model_selection import train_test_split

from sklearn import linear_model

#from sklearn.model_selection import train_test_split, KFold

from statsmodels.tsa.stattools import adfuller

#MSE
from sklearn.metrics import mean_squared_error
#For square rooting
from math import sqrt
#For implementing SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX

#For importing the drop-down placed at LinearForm
from .forms import LinearForm, UploadFileForm, IntForm, p_q, p_d_q, p_d_q_s, AlphaForm, DoubleForm, TripleForm

from statsmodels.tsa import seasonal
#from django.http import JsonResponse
#from . import utils

# Similar to auto.arima in R
from pyramid.arima import auto_arima

#Implementing ARMA and ARIMA with statsmodels
from statsmodels.tsa.arima_model import ARMA, ARIMA


from fusioncharts import FusionCharts #used to project the forecast outcomes

import os # for file reading used

#rendering the output onto the graphs
from django.shortcuts import render
from django.template import loader

#below package is used for storing the uploaded file into db as a reference
from django.core.files.storage import FileSystemStorage

# Create your views here.

#Home method is for uploading the data
def home(request):

    template_name = 'polls/home.html'
    #UploadFileForm is responsible for the form method
    contents = UploadFileForm(request.POST, request.FILES)

    if request.method == "POST":

        global data, upload_file_url

        if contents.is_valid():

            file_dict = (contents.cleaned_data['Upload_File'])

            fs = FileSystemStorage()

            global file_dict_name, upload_file_url

            file_dict_name = file_dict.name
            #.save saves the file
            my_file = fs.save(file_dict.name, file_dict)

            upload_file_url = fs.url(my_file)

            data = display_data(file_dict_name)

            #After performing the above operation data now contains the data
            return render(request, template_name, {'upload_file_url': upload_file_url, 'cols': str(data.columns.values)} )

    return render(request, template_name, {'contents' : contents})


#to select specific columns as target variable
def select_column(request):

    template_name = 'polls/home.html'

    global column_name, data_clean_

    _data_column_names = (data.columns.values)

    a = b = tuple(_data_column_names)

    disp_cols = tuple(zip(a,b))

    form = LinearForm(request.POST, choices = disp_cols)

    if request.method == "POST" and form.is_valid():

        data_clean_ = (form.cleaned_data['choice_field'])

        column_name = data_clean_

        column_name = column_proper_format(_data_column_names)

        args = {'column_name': data_clean_,
                'cols': column_proper_format(_data_column_names),
                'form': form}

        return render(request, 'polls/home.html', args)

    column_name = column_proper_format(_data_column_names)

    args = {'cols': column_name,
            'Masking': 'Masking',
                    'form': form}

    return render(request, template_name, args)


#To visualize the data columns in a proper format #Not used!
def column_proper_format(_data_column_names):

    sep_join = ", "

    column_name = sep_join.join(_data_column_names)

    return column_name



#used for reading the data and backup_data is for the backup of the data
def display_data(file_dict_name):

    di = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'media')

    path = os.path.join(di, file_dict_name)

    global data, backup_data

    filename, file_extension = os.path.splitext(path)

    if file_extension == '.csv':

        data = pd.read_csv(path, mangle_dupe_cols= True)
        backup_data = data

        return proper_format(data, path, file_extension)

    elif file_extension == '.json':

        data = pd.read_json(path, mangle_dupe_cols= True)
        backup_data = data

        return proper_format(data, path, file_extension)

    elif file_extension == '.xls':

        data = pd.read_excel(path, mangle_dupe_cols= True)
        backup_data = data

        return proper_format(data)


#getting the data in proper format
def proper_format(data, path, file_extension):

    global original, dates, i

    reference = ['Date', 'month', 'year' ,'time', 'Month', 'date']

    for i in data.columns.values:

        if i in reference:

            dates = (data.ix[0:,i])

            data.index.name = i

            data.index = date_format(dates)

            data = data.drop([i], axis = 1)

            break

        else:

            data.index = pd.Series([i for i in data.index])

            data.index.name = "Numeric column"

    original =  data

    return data


#getting date in suitable time format
def date_format(dates_):

    try:

        dates_ =pd.to_datetime(dates_, format='%Y-%m-%d')

    except ValueError:

        dates_ =pd.to_datetime(dates_, format='%m-%d')

    except ValueError:

        dates_ = pd.to_datetime(dates_)

    return dates_


#Making the series Stationary
def stationarity(request):

    template_name = 'polls/Components.html'

    _data_ = data

    dfoutput = checking_stationarity(data_clean_, _data_)

    if dfoutput['outcome'] == 'Yes':

        #data_station = np.log(data[data_clean_])

        #data_station = pd.DataFrame(data_station)

        yo = pd.DataFrame(data.ix[:,data_clean_])

        output = lines_markers(yo)

        output = {'output': output}

        return render(request, template_name, output)

    else:


        data_station = pd.DataFrame(_data_[data_clean_])

        dfoutput = checking_stationarity(data_clean_, data_station)

        data_station = np.log(_data_[data_clean_])

        moving_avg = pd.rolling_mean(data_station,12)

        output = pd.DataFrame(data_station - moving_avg)

        output = lines_markers(output, title_name = "Stationarity")

        output = {'output': output}

        return render(request, template_name, output)

    return render(request, template_name, )

#Checking whether series is stationary or not
def checking_stationarity(selected_column, _data_):

    result = adfuller(_data_.ix[:,selected_column])

    dfoutput = pd.Series(result[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in result[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    dfoutput = dfoutput.round(decimals=3)

    if result[0]> result[4]['5%']:

        outcome = 'No'

    else:
        outcome = 'Yes'

    dfoutput['outcome'] = outcome

    return dfoutput

#plotting for Stationarity
def stationary_out(request):

    template_name = 'polls/other_output.html'
    data_ = data
    dfoutput = checking_stationarity(data_clean_, data_)

    tracey = list()

    for i in dfoutput.index.values:
        print i
        pmet = dfoutput[i]
        tracey.append(pmet)

    trace = go.Table(
        header=dict(values=[i for i in dfoutput.index],
                    line = dict(color='#7D7F80'),
                    fill = dict(color='#a1c3d1'),
                    align = ['center'] * len(dfoutput)),
        cells=dict(values=[tracey[i] for i in range(0, len(dfoutput.index))],
                   line = {'color' : '#7D7F80'},
                   fill = {'color' : '#EDFAFF'},
                   align = ['center'] * len(dfoutput)))

    tempo = go.Data([trace])

    layout= go.Layout(autosize = True, width = 900, height = 1100,
                       margin=go.Margin(
        l=50,
        r=50,
        b=150,
        t=100,
        pad=4

    ), title="Checking for Stationarity", xaxis={'title':'X'}, yaxis={'title':'Y'})

    figure=go.Figure(data = tempo,layout = layout)

    output = opy.plot(figure, auto_open = True, output_type='div')

    output = {'output': output}

    return render(request, template_name, output)

#Moving Averages
def moving_averages(request):

    template_name = 'polls/Graph_output_NSM.html'

    data_ = data.ix[:, data_clean_]

    #original is already defined in proper_format as global variable
    original =  data.ix[:len(data)-5, data_clean_]

    #determine the window value
    modified = pd.rolling_mean(original, window = 12)

    modified = modified.dropna()

    rms = sqrt(mean_squared_error(data.ix[len(data)-5:,data_clean_], modified[:5]))

    y_actuals = data.ix[len(data)-5:, data_clean_].values

    y_hat = modified[:5].values

    mape = mean_absolute_percentage_error(y_actuals, y_hat)

    forecasts = modified[len(modified)-12:]

    data_ = data.ix[:, data_clean_]

    seriesname = "MA"

    output = graphing1(data_, forecasts, rms, mape, seriesname, _values_ = list([None, None, None]))

    return render(request, template_name, output)

#Exponential
def exponential_smoothing(request):

    '''Exponential Smoothing'''

    template_name = 'polls/Simple_expo_middle.html'

    alpha = 0.1 #smoothing parameter

    original = data.ix[:len(data)-5, data_clean_]

    list_original = list(data.ix[:len(data)-5, data_clean_])

    modified = [original[0]] # first value is same as in the series

    for n in range(1, len(original)+17):
        # loc of 259 is in the format of the following
        # yhat = alpha * current data point + (1 - alpha) * (previous expected datapoint)
        modified.append(alpha * list_original[n] + (1 - alpha) * modified[n-1])

        list_original.append(alpha * list_original[n] + (1 - alpha) * modified[n-1])

    modified = np.array(modified[len(original): ])

    rms = sqrt(mean_squared_error(data.ix[len(data)-5:, data_clean_], modified[:5]))

    y_actuals = data.ix[len(data)-5:, data_clean_]

    y_hat = modified[:5]
    #for calculating mape
    mape = mean_absolute_percentage_error(y_actuals, y_hat)

    original = data.ix[:,data_clean_]

    modified = modified[:12]

    _values_ = list([alpha, None, None])

    seriesname = "Simple Exponential Smoothing"

    output = graphing1(original, modified, rms, mape, seriesname, _values_)

    return render(request, template_name, output)

#Simple Exponential parameters from front end
def exponential_smoothing_manual(request):

    template_name = 'polls/Simple_expo_manual.html'

    form = AlphaForm(request.POST)

    if request.method == "POST":

        if form.is_valid():

            alpha = form.cleaned_data['alpha']

            original = data.ix[:len(data)-5, data_clean_]

            list_original = list(data.ix[:len(data)-5, data_clean_])

            modified = [original[0]] # first value is same as in the series

            for n in range(1, len(original)+17):
                # loc of 259 is in the format of the following
                # yhat = alpha * current data point + (1 - alpha) * (previous expected datapoint)
                modified.append(alpha * list_original[n] + (1 - alpha) * modified[n-1])

                list_original.append(alpha * list_original[n] + (1 - alpha) * modified[n-1])

            modified = np.array(modified[len(original): ])

            rms = sqrt(mean_squared_error(data.ix[len(data)-5:, data_clean_], modified[:5]))

            y_actuals = data.ix[len(data)-5:, data_clean_]

            y_hat = modified[:5]
            #for calculating mape
            mape = mean_absolute_percentage_error(y_actuals, y_hat)

            original = data.ix[:,data_clean_]

            modified = modified[:12]

            _values_ = list([alpha, None, None])

            seriesname = "Simple Exponential Smoothing"

            output = graphing1(original, modified, rms, mape, seriesname, _values_)

            return render(request, template_name, output)

    return render(request, template_name, {'form':form})



#double exponential
def double_exponential_smoothing(request):

    template_name = 'polls/Double_expo_middle.html'

    alpha, beta = 0.1, 0.9#smoothing parameters

    original = data.ix[:len(data)-5, data_clean_]

    modified = [original[0]]

    level = 0

    trend = 0

    for n in range(0, len(original)+16):

        if n == 1:

            trend = original[1] - original[0]

            level  = original[0]

        if n >= len(original): # we are forecasting

          value = modified[-1]

        else:

          value = original[n]

        last_level, level = level, alpha*value + (1-alpha)*(level+trend)

        trend = beta*(level-last_level) + (1-beta)*trend

        modified.append(level+trend)

    modified = np.array(modified[len(original):])

    rms = sqrt(mean_squared_error(data.ix[len(data)-5:, data_clean_], modified[:5]))

    modified = modified[:12]

    original = data.ix[:, data_clean_]

    y_actuals = data.ix[len(data)-5:, data_clean_]

    y_hat = modified[:5]
    #calculating mape
    mape = mean_absolute_percentage_error(y_actuals, y_hat)

    seriesname = "Double Exponential Smoothing"

    _values_ = list([alpha, beta, None])

    output = graphing1(original, modified, rms, mape, seriesname, _values_)

    return render(request, template_name, output)

#Double exponential feeding parameters from front end
def double_exponential_smoothing_manual(request):

    template_name = 'polls/Double_expo_manual.html'

    form = DoubleForm(request.POST)

    if request.method == "POST" and form.is_valid():

        alpha = form.cleaned_data['alpha']

        beta = form.cleaned_data['beta']

        original = data.ix[:len(data)-5, data_clean_]

        modified = [original[0]]

        level = 0

        trend = 0

        for n in range(0, len(original)+12):

            if n == 1:

                trend = original[1] - original[0]

                level  = original[0]

            if n >= len(original): # we are forecasting

              value = modified[-1]

            else:

              value = original[n]


            last_level, level = level, alpha*value + (1-alpha)*(level+trend)

            trend = beta*(level-last_level) + (1-beta)*trend

            modified.append(level+trend)

        modified = np.array(modified[len(original):])

        rms = sqrt(mean_squared_error(data.ix[len(data)-5:, data_clean_], modified[:5]))

        modified = modified[:12]

        original = data.ix[:, data_clean_]

        y_actuals = data.ix[len(data)-5:, data_clean_]

        y_hat = modified[:5]
        #calculating mape
        mape = mean_absolute_percentage_error(y_actuals, y_hat)

        seriesname = "Double Exponential Smoothing"

        _values_ = list([alpha, beta, None])

        output = graphing1(original, modified, rms, mape, seriesname, _values_)

        return render(request, template_name, output)

    return render(request, template_name, )

#Triple exponential
def triple_exponential_smoothing(request):

    template_name = 'polls/Triple_expo_middle.html'

    series = data.ix[:len(data), data_clean_]

    original_ = data.ix[:len(data), data_clean_]

    alpha, beta, gamma = 0.71, 0.02, 0.99

    def initial_trend(series, slen):

        _sum_ = 0.0

        for i in range(slen):

            _sum_ += float(series[i+slen] - series[i]) / slen

        return _sum_ / slen

    def initial_seasonal_components(series, slen):

        seasonals = {}

        season_averages = []

        n_seasons = int(len(series)/slen)

        # compute season averages

        for j in range(n_seasons):

            season_averages.append(sum(series[slen*j : slen * j + slen])/(float(slen)))

        # compute initial values

        for i in range(slen):

            sum_of_vals_over_avg = 0.0

            for j in range(n_seasons):

                sum_of_vals_over_avg = sum_of_vals_over_avg + series[slen*j+i]-season_averages[j]

                seasonals[i] = sum_of_vals_over_avg/n_seasons

        return seasonals


    def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):

        result = []

        seasonals = initial_seasonal_components(series, slen)

        for i in range(len(series)+n_preds):

            if i == 0: # initial values

                smooth = series[0]

                trend = initial_trend(series, slen)

                result.append(series[0])

                continue

            if i >= len(series): # we are forecasting

                m = i - len(series) + 1

                result.append((smooth + m*trend) + seasonals[i%slen])

            else:

                val = series[i]

                last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)

                trend = beta * (smooth-last_smooth) + (1-beta)*trend

                seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]

                result.append(smooth+trend+seasonals[i%slen])

        return result

    modified = triple_exponential_smoothing(original_, 3, alpha, beta, gamma, 17)

    modified = np.array(modified[len(series):])

    rms = sqrt(mean_squared_error(data.ix[len(data)-5:, data_clean_], modified[:5]))

    modified = modified[:12]

    original = data.ix[:, data_clean_]

    y_actuals = original.ix[len(data)-5:,]

    y_hat = modified[:5]
    #calculating mape
    mape = mean_absolute_percentage_error(y_actuals, y_hat)

    _values_ = list([alpha, beta, gamma])

    seriesname = "Holt-Winters"

    output = graphing1(original_, modified, rms, mape,  seriesname, _values_)

    return render(request, template_name, output)

#Triple exponential values from front end
def triple_exponential_smoothing_manual(request):

    template_name = 'polls/Triple_expo_manual.html'

    form = TripleForm(request.POST)

    if request.method == "POST" and form.is_valid():

        alpha = form.cleaned_data['alpha']
        beta = form.cleaned_data['beta']
        gamma = form.cleaned_data['gamma']

        series = data.ix[:len(data), data_clean_]

        original_ = data.ix[:len(data), data_clean_]


        def initial_trend(series, slen):

            sum = 0.0

            for i in range(slen):

                sum += float(series[i+slen] - series[i]) / slen

            return sum / slen

        def initial_seasonal_components(series, slen):

            seasonals = {}

            season_averages = []

            n_seasons = int(len(series)/slen)

            # compute season averages

            for j in range(n_seasons):

                season_averages.append(sum(series[slen*j : slen * j + slen])/(float(slen)))

                # compute initial values

            for i in range(slen):

                sum_of_vals_over_avg = 0.0

                for j in range(n_seasons):

                    sum_of_vals_over_avg = sum_of_vals_over_avg + series[slen*j+i]-season_averages[j]

                    seasonals[i] = sum_of_vals_over_avg/n_seasons

            return seasonals


        def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):

            result = []

            seasonals = initial_seasonal_components(series, slen)

            for i in range(len(series)+n_preds):

                if i == 0: # initial values

                    smooth = series[0]

                    trend = initial_trend(series, slen)

                    result.append(series[0])

                    continue

                if i >= len(series): # we are forecasting

                    m = i - len(series) + 1

                    result.append((smooth + m*trend) + seasonals[i%slen])

                else:

                    val = series[i]

                    last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)

                    trend = beta * (smooth-last_smooth) + (1-beta)*trend

                    seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]

                    result.append(smooth+trend+seasonals[i%slen])

            return result


        modified = triple_exponential_smoothing(original_, 3, alpha, beta, gamma, 17)

        modified = np.array(modified[len(series):])

        rms = sqrt(mean_squared_error(data.ix[len(data)-5:, data_clean_], modified[:5]))

        modified = modified[5:]

        original = data.ix[:, data_clean_]

        y_actuals = original.ix[len(data)-5:,]

        y_hat = modified[:5]
        #calculating mape
        mape = mean_absolute_percentage_error(y_actuals, y_hat)

        _values_ = list([alpha, beta, gamma])

        seriesname = "Holt-Winters"

        output = graphing1(original_, modified, rms, mape,  seriesname, _values_)

        return render(request, template_name, output)

    return render(request, template_name, )

#used for viewing the components of the time series
def graphical_v(graph_data, sub_caption, dates):

  # Create an object for the Area 2D chart using the FusionCharts class constructor
    msbarChart = FusionCharts("msline", "ex1" , "950", "450", "chart-1", "json",
		# The chart data is passed as a string to the `dataSource` parameter.
      {
      "chart": {
        "caption":  "Number of {} on Y-axis {} on X-axis" .format(data.columns.values[0], data.index.names[0]),
         "subcaption": sub_caption,
          "linethickness": "2",
          "showvalues": "0",
          "formatnumberscale": "1",
          "labeldisplay": "ROTATE",
          "slantlabels": "1",
          "divLineAlpha": "40",
          "anchoralpha": "0",
          "animation": "1",
          "legendborderalpha": "40",
          "drawCrossLine": "1",
          "crossLineColor": "#0d0d0d",
          "crossLineAlpha": "100",
          "tooltipGrayOutColor": "#80bfff",
          "yaxisminValue":min(graph_data)-5,
          "yaxismaxValue":max(graph_data)+5,
          "animation":"1",
          "theme": "fint",
      },

      "categories": [{
        "category": [[{'label' : i} for i in dates],
                     ]
      },
    ],
      "dataset": [{
        "seriesname": sub_caption,
        "data": [[{'value': i} for i in graph_data],
                 ]
      }]
    })

	# Alternatively, you can assign this string to a string variable in a separate JSON file and
	# pass the URL of that file to the `dataSource` parameter.

    output = {'output' : msbarChart.render()
              }

    return  output


#taking out residual component with seasonal_decompose
def Resid_component(request):

    '''Displaying the original series on the chart'''

    template_name = 'polls/Components.html'

    #this_temp_freq, this_temp_series_components

    try:

        if this_temp_freq != None:

            Resid_component = this_temp_series_components.resid

            Resid_component = pd.Series.dropna(Resid_component)

            sub_caption = "Random component"

            #output = graphical_v(Resid_component, sub_caption, Resid_dates)

            output = lines_markers(pd.DataFrame(Resid_component), sub_caption)

            output = {'output': output}

            return render(request, template_name, output)

    except NameError:

        form = IntForm(request.POST)

        if request.method == "POST" and form.is_valid():

            frequency = form.cleaned_data['frequency']

            time_series_components = seasonal.seasonal_decompose(data.ix[0:,data_clean_], freq= frequency)

            Resid_component = time_series_components.resid

            #Resid_dates = dates#[~Resid_component.isnull()]

            Resid_component = pd.Series.dropna(Resid_component)

            sub_caption = "Random component"

            #output = graphical_v(Resid_component, sub_caption, Resid_dates)

            output = lines_markers(pd.DataFrame(Resid_component), sub_caption)

            output = {'output': output}

            #args = {'output': output, 'form_freq': form_freq}

            return render(request, template_name, output)

    return render(request, template_name, {'form': form})


#taking out Trend component with seasonal_decompose

def Trend_component(request):

    '''Displaying the original series on the chart'''

    global time_series_components, frequency

    template_name = 'polls/Components.html'

    form = IntForm(request.POST)

    if request.method == "POST" and form.is_valid():

        frequency = form.cleaned_data['frequency']

        global this_temp_freq, this_temp_series_components

        this_temp_freq = frequency

        time_series_components = seasonal.seasonal_decompose(data.ix[0:,data_clean_], freq= frequency)

        this_temp_series_components = time_series_components

        Trend_component = time_series_components.trend

        #Trend_dates = dates

        Trend_component = pd.Series.dropna(Trend_component)

        sub_caption = "Trend component"

        #output = graphical_v(Trend_component, sub_caption, Trend_dates)
        output = lines_markers(pd.DataFrame(Trend_component), sub_caption)

        output = {'output': output}
        #args = {'output': output, 'form_freq': form_freq}

        return render(request, template_name, output)

    return render(request, template_name, {'form': form})

#taking out Seasonal component with seasonal_decompose
def Seasonal_component(request):

    '''Displaying the original series on the chart'''

    template_name = 'polls/Components.html'

    #global time_series_components, frequency

    try:

        if this_temp_freq != None:

            Seasonal_component = this_temp_series_components.seasonal

            sub_caption = "Seasonal component"

            #output = graphical_v(Seasonal_component, sub_caption, Seasonal_dates)
            output = lines_markers(pd.DataFrame(Seasonal_component), sub_caption)

            output = {'output': output}

            return render(request, template_name, output)

    except NameError:

        form = IntForm(request.POST)

        if request.method == "POST" and form.is_valid():

            frequency = form.cleaned_data['frequency']

            time_series_components = seasonal.seasonal_decompose(data.ix[0:,data_clean_], freq= frequency)

            Seasonal_component = time_series_components.seasonal

            #Seasonal_dates = dates

            #Seasonal_component = pd.Series.dropna(Trend_component)

            sub_caption = "Seasonal component"

            #output = graphical_v(Seasonal_component, sub_caption, Seasonal_dates)
            output = lines_markers(pd.DataFrame(Seasonal_component), sub_caption)

            output = {'output': output}

            #args = {'output': output, 'form_freq': form_freq}

            return render(request, template_name, output)

    return render(request, template_name, {'form': form})


#ARMA model
def ARMA_model(request):
    '''Auto Regressive Moving Average'''

    output = ARMA_auto()

    return render(request, 'polls/ARMA_auto.html', output)


#ARMA auto_arima
def ARMA_auto():

    #template_name = 'polls/Graph_ouput.html'

    data_ = data.ix[:len(data)-5, data_clean_]

    model = auto_arima(data_, start_p=1, start_q=1, max_p=3, max_q=3, m=1,
                        seasonal=False, n_jobs=-1, d=0, D=1, trace=True,
                        error_action='ignore',  # don't want to know if an order does not work
                        suppress_warnings=True,  # don't want convergence warnings
                        stepwise=True, random=True, random_state=42,  # we can fit a random search (not exhaustive)
                        n_fits=25)


    model1 = ARMA(data_.astype('float64'),order = model.order).fit()


    temp2 = (model1.forecast(17))
    forecasts = np.array(temp2[0])
    cofidence_interval = (temp2[2]) # CONFIDENCE INTERVAL

    lower, upper = list(), list()

    temp = np.array(cofidence_interval)

    for i in range(len(temp)):

        lower.append(temp[i,0])

        upper.append(temp[i,1])

    rms = sqrt(mean_squared_error(data.ix[len(data)-5:, data_clean_], forecasts[:5]))

    y_actuals = data.ix[len(data)-5:, data_clean_]

    y_hat = forecasts[:5]
    #mean_absolute_percentage_error
    mape = mean_absolute_percentage_error(y_actuals, y_hat)

    data_ = data.ix[:, data_clean_]

    forecasts = forecasts[:12]

    _values_ = list(model.order)

    seriesname = "ARMA"

    output = graphing2(data_, forecasts, cofidence_interval, rms, mape, seriesname, _values_)

    #return render(request, template_name, output)
    return output


def arma_manual(request):

    template_name = 'polls/ARMA_manual.html'

    form = p_q(request.POST)
        #q = p_q(request.POST)

    if request.method == "POST" and form.is_valid():

        p = form.cleaned_data['p']
        q = form.cleaned_data['q']

        data_ = data.ix[:len(data)-5, data_clean_]

        model1 = ARMA(data_.astype('float64'),order = (p,q)).fit()


        temp2 = (model1.forecast(17))
        forecasts = np.array(temp2[0])
        cofidence_interval = (temp2[2]) # CONFIDENCE INTERVAL

        lower, upper = list(), list()

        temp = np.array(cofidence_interval)

        for i in range(len(temp)):

            lower.append(temp[i,0])

            upper.append(temp[i,1])

        rms = sqrt(mean_squared_error(data.ix[len(data)-5:, data_clean_], forecasts[:5]))

        y_actuals = data.ix[len(data)-5:, data_clean_]

        y_hat = forecasts[:5]
        #mean_absolute_percentage_error
        mape = mean_absolute_percentage_error(y_actuals, y_hat)

        data_ = data.ix[:, data_clean_]

        forecasts = forecasts[:12]

        _values_ = [p, 0, q]

        seriesname = "ARMA"

        output = graphing2(data_, forecasts, cofidence_interval, rms, mape, seriesname, _values_)

        return render(request, template_name, output)

    return render(request, template_name, )


#ARIMA model
def arima2(request):

    return render(request, 'polls/Graph_ouput_ARIMA.html', )

def arima_auto(request):

    template_name = 'polls/Graph_ouput_ARIMA.html'

    '''Auto Regressive Integrated Moving Average'''
    data_ = data.ix[:, data_clean_]

    model = auto_arima(data_, start_p=1, start_q=1, max_p=3, max_q=3, m=1,
                    seasonal=False, n_jobs=-1, d=1,D=1, trace=True,
                    error_action='ignore',  # don't want to know if an order does not work
                    suppress_warnings=True,
                    stepwise=True, random=False#, random_state=42,  # we can fit a random search (not exhaustive)
                    #n_fits=25
                    )

    #throws an error if not float64
    model1 = ARIMA(data_.astype('float64'),order = model.order).fit()

    temp2 = (model1.forecast(17, alpha = 0.05))

    forecasts = np.array(temp2[0])

    cofidence_interval = (temp2[2]) # CONFIDENCE INTERVAL

    rms = sqrt(mean_squared_error(data.ix[len(data)-5:, data_clean_], forecasts[:5]))

    y_actuals = data.ix[len(data)-5:, data_clean_]

    y_hat = forecasts[:5]
    #for calculating mape
    mape = mean_absolute_percentage_error(y_actuals, y_hat)

    lower, upper = list(), list()

    temp = np.array(cofidence_interval)

    for i in range(len(temp)):

        lower.append(temp[i,0])
        upper.append(temp[i,1])

    #confidence_intervals = np.concatenate((np.array(lower),np.array(upper)))

    data_ = data.ix[:,data_clean_]

    forecasts = forecasts[:12]

    seriesname = "ARIMA2"

    _values_ = [i for i in model.order]

    output = graphing2(data_, forecasts, cofidence_interval, rms,  mape, seriesname, _values_)

    return render(request, template_name, output)


def arima_manual(request):

    template_name = 'polls/ARIMA_manual.html'

    if request.method == "POST":

        form = p_d_q(request.POST)

        if form.is_valid():

            p = form.cleaned_data['p']
            d = form.cleaned_data['d']
            q = form.cleaned_data['q']

            data_ = data.ix[:len(data)-5, data_clean_]

            model1 = ARIMA(data_.astype('float64'),order = (p,d,q)).fit()

            temp2 = (model1.forecast(17, alpha = 0.05))

            forecasts = np.array(temp2[0])

            cofidence_interval = (temp2[2]) # CONFIDENCE INTERVAL

            rms = sqrt(mean_squared_error(data.ix[len(data)-5:, data_clean_], forecasts[:5]))

            y_actuals = data.ix[len(data)-5:, data_clean_]

            y_hat = forecasts[:5]
            #for calculating mape
            mape = mean_absolute_percentage_error(y_actuals, y_hat)

            lower, upper = list(), list()

            temp = np.array(cofidence_interval)

            for i in range(len(temp)):

                lower.append(temp[i,0])
                upper.append(temp[i,1])

            #confidence_intervals = np.concatenate((np.array(lower),np.array(upper)))

            data_ = data.ix[:,data_clean_]

            forecasts = forecasts[5:]

            _values_ = [p,d,q]

            seriesname = "ARIMA"

            output = graphing2(data_, forecasts, cofidence_interval, rms,  mape, seriesname, _values_)

            return render(request, template_name, output)

    return render(request, template_name, )

def dating(temp1):
    global yearing

    yearing, b, final =list(), list(), list()

    if len(temp1[0])< 7:

        a = (int((temp1[len(temp1.index)-1])[0]) + 1)

        a = str(a)

        for i in range(0, len(data), 12):

            yearing.append(str(temp1[i])[0])


        for i in range(1,13):

            b.append(int((temp1[len(temp1.index)-i])[1:]))

            final.append(a + str(b[i-1]))

        final.reverse()

    else:

        a = str(int(temp1[len(temp1.index)-1][0:4])+1)

        for i in range(0, len(data), 12):

            yearing.append(str(temp1[i])[:4])

        for i in range(1,13):

            b.append(str(temp1[len(temp1.index)-i])[4:8])

            final.append(a+str(b[i-1]))

        final.reverse()

    return final

#Naive, Simple, Exponential, and DOuble methods use this graphing function
def graphing1(original, modified, rmse,  mape,seriesname,  _values_):
  #Create an object for the Area 2D chart using the FusionCharts class constructor

    len_mod = len(modified)
    #modified has forecasted values

    future_dates = (dating(dates))

    #future_dates = future_dates.reverse()

  #original has original dataset values
    msbarChart = FusionCharts("msline", "ex1" , "950", "650", "chart-1", "json",
		# The chart data is passed as a string to the `dataSource` parameter.
      {
      "chart": {
        "caption": "Number of {} on the Y-axis and {} on X-axis" .format(data_clean_, data.index.names[0]),
         "subcaption": "Model used: {} RMSE score {} MAPE score {} Alpha: {} Beta: {} Gamma: {}" .format(seriesname, round(rmse, 3), mape, _values_[0], _values_[1], _values_[2]),
          "linethickness": "2",
          "adjustDiv": "0",
          "showvalues": "0",
          "baseFontSize" : "14",
          "formatnumberscale": "0",
          "labeldisplay": "ROTATE",
          "enableiconmousecursors": "1",
          "axis":"linear",
          "dynamicaxis": "1",
          "slantlabels": "1",
          "divLineAlpha": "40",
          "anchoralpha": "0",
          "animation": "1",
          "legendborderalpha": "20",
          "drawCrossLine": "1",
          "crossLineColor": "#0d0d0d",
          "crossLineAlpha": "100",
          "tooltipGrayOutColor": "#80bfff",
          "yaxisminValue":int(min(original))-5,
          "yaxismaxValue":int(max(original))+5,
          "animation":"1",
          "theme": "fint",
      },
      "categories": [{
        "category": [[{'label' : i} for i in dates],
                     [{'label': i} for i in future_dates]
                     ]
      },
    ],
      "dataset": [{
        "seriesname": "Original data",
        "data": [[{'value': i} for i in original],
                  [{'value': ' '} for x in range(len_mod)]]
      }, {
        "seriesname": "Forecast",
        "data": [[{'value': ' '} for i in range(len(dates))],
                 [{'value' : i} for i in modified]]
        }

     ]
    })

    output = {'output' : msbarChart.render()
              }

    return  output

#Naive method implemenation
def Naive_method(request):

    template_name = 'polls/Graph_output_NSM.html'

    original = data.ix[len(data)-5:, data_clean_]

    dd = np.asarray(data)

    y_hat = list()

    for i in range(1,18):

        y_hat.append(dd[len(dd) - i])

    y_hat = list(reversed(y_hat))

    y_hat = pd.DataFrame(y_hat)

    forecast = y_hat.ix[:17,0]

    rmse = sqrt(mean_squared_error(data.ix[len(data)-5:, data_clean_], forecast[:5]))

    y_actuals = data.ix[len(data)-12:, data_clean_].values

    y_hat = forecast[5:]
    #for calculating mape
    mape = mean_absolute_percentage_error(y_actuals, y_hat)

    original = data.ix[:, data_clean_]

    _values_ = list([None, None, None])
    seriesname = "Naive"
    output = graphing1(original, forecast[5:], rmse, mape, seriesname, _values_)

    return render(request, template_name , output)


def sarimax_multi(request):

    template_name = 'polls/SARIMAX_middle.html'

    global data_, text, exog_var, endog_var

    #template_name = 'polls/ARIMAX_middle.html'

    data_ = data

    text = str()

    _data_column_names = (data.columns.values)

    a = tuple(_data_column_names)

    b = tuple(_data_column_names)

    disp_cols = tuple(zip(a,b))

    form = LinearForm(request.POST, choices = disp_cols)

    args = {'form': form, 'text': text}

    if request.method == "POST":

        if form.is_valid():

            text = form.cleaned_data['choice_field']

            exog_var = (data_.drop(text, axis=1))#dependencies

            endog_var = data_.ix[:,text]#y_hat

            model = auto_arima(endog_var, exogenous = exog_var ,start_p=1, start_q=1, max_p=3, max_q=3, m=1,
                    start_P=0, seasonal=True, n_jobs=-1, d=1, D=0, trace=True,
                    error_action='ignore',
                    suppress_warnings=True,  # don't want convergence warnings
                    stepwise=True, random=True, random_state=42,  # we can fit a random search (not exhaustive)
                    n_fits=25)

            a = SARIMAX(endog_var.astype('float64'), order = model.order, exog = exog_var, seasonal_order = model.seasonal_order).fit()
            model_order = [i for i in model.order]
            seasonal_order = [i for i in model.seasonal_order]

            #_values_ = model_order+seasonal_order

            modified = a.predict(start = len(exog_var)-12, dyanmic = True)

            y_actuals = data.ix[len(data)-5:, text].values

            y_hat = modified[:5]

            mape = mean_absolute_percentage_error(y_actuals, y_hat)

            confidence = None

            rms = sqrt(mean_squared_error(data.ix[len(data)-5:,text], modified[:5]))

            seriesname = "SARIMAX"

            def mean_confidence_interval(modified, confidence=0.95):

                mod_arry = 1.0 * np.array(modified.dropna())

                n = len(mod_arry)

                m, se = np.mean(mod_arry), scipy.stats.sem(mod_arry)

                h = se * sp.stats.t._ppf((1+confidence)/2., n-1)

                return m-h, m+h

            lower, upper = list(), list()

            lower_val, upper_val = mean_confidence_interval(modified, confidence=0.95)

            for i in range(len(modified)):

                lower.append(modified.ix[i,0] - lower_val)
                #lower.append(modified.ix[i,0])
                upper.append(modified.ix[i,0] + upper_val)
                #upper.append(modified.ix[i,0])

            confidence = np.concatenate((np.array(lower),np.array(upper)))

            _original_ =  data.ix[:,text]

            output = graphing3(_original_, modified, confidence, rms, mape, seriesname, text, model_order, seasonal_order)

            args = {'form': form, 'output': output}

            return render(request, template_name, output)

    args = {'form': form, 'text': 'Pass the variable'}

    return render(request, 'polls/SARIMAX_output.html', args)

def sarimax_manual(request):

    template_name = 'polls/SARIMAX_man.html'

    if request.method == "POST":

        form = p_d_q_s(request.POST)

        if form.is_valid():


            p = form.cleaned_data['p']
            d = form.cleaned_data['d']
            q = form.cleaned_data['q']

            P = form.cleaned_data['P']
            D = form.cleaned_data['D']
            Q = form.cleaned_data['Q']
            s = form.cleaned_data['s']

            data_ = data

            template_name = 'polls/SARIMAX_man.html'

            exog_var = (data_.drop(text, axis=1))#dependencies

            endog_var = data_.ix[:,text]#y_hat

            a = SARIMAX(endog_var.astype('float64'), order = (p,d,q), exog = exog_var, seasonal_order = (P, D, Q,s)).fit()

            model_order = [p,d,q]

            seasonal_order = [P, D, Q, s]

            modified = a.predict(start = len(exog_var)-12, dyanmic = True)

            y_actuals = data.ix[len(data)-5:, text].values

            y_hat = modified[:5]

            mape = mean_absolute_percentage_error(y_actuals, y_hat)

            confidence = None

            rms = sqrt(mean_squared_error(data.ix[len(data)-5:,text], modified[:5]))

            seriesname = "SARIMAX"

            def mean_confidence_interval(modified, confidence=0.95):

                mod_arry = 1.0 * np.array(modified.dropna())

                n = len(mod_arry)

                m, se = np.mean(mod_arry), scipy.stats.sem(mod_arry)

                h = se * sp.stats.t._ppf((1+confidence)/2., n-1)

                return m-h, m+h

            lower, upper = list(), list()

            lower_val, upper_val = mean_confidence_interval(modified, confidence=0.95)

            for i in range(len(modified)):

                lower.append(modified.ix[i,0] - lower_val)
                upper.append(modified.ix[i,0] + upper_val)

            confidence = np.concatenate((np.array(lower),np.array(upper)))

            _original_ =  data.ix[:,text]

            output = graphing3(_original_, modified, confidence, rms, mape, seriesname, text, model_order, seasonal_order)

            #args = {'form': form, 'output': output}

            return render(request, template_name, output)

    #args = {'form': form, 'text': 'Pass the variable'}

    return render(request, template_name , )


#ARIMAX
def arimax(request):

    global data_, text, exog_var, endog_var

    template_name = 'polls/ARIMAX_middle.html'

    data_ = data

    text = str()

    _data_column_names = (data.columns.values)

    a = tuple(_data_column_names)

    b = tuple(_data_column_names)

    disp_cols = tuple(zip(a,b))

    form = LinearForm(request.POST, choices = disp_cols)

    args = {'form': form, 'text': text}

    if request.method == "POST" and form.is_valid():

        text = form.cleaned_data['choice_field']

        exog_var = (data_.drop(text, axis=1))#dependencies

        endog_var = data_.ix[:,text]#y_hat

        model = auto_arima(endog_var, exogenous = exog_var ,start_p=1, start_q=1, max_p=3, max_q=3, m=1,
                    seasonal = False, n_jobs=-1, d=1, trace=True,
                    error_action='ignore',
                    suppress_warnings=True,  # don't want convergence warnings
                    stepwise=True, random=True, random_state=42,  # we can fit a random search (not exhaustive)
                    n_fits=25)

        a = ARIMA(endog_var.astype('float64'), order = model.order, exog = exog_var).fit()

        model_order = [i for i in model.order]
        seasonal_order = "Not applicable"

        modified = list()

        for i in range(0,13):

            temp = exog_var.ix[len(data_) - i: ]

        modified = a.forecast(steps = i, exog = temp.values)

        in_sample_pred = a.predict(start = len(exog_var) - 5 )

        y_actuals = data.ix[len(data)-5:, text].values

        y_hat, confidence = modified[0], modified[2]

        mape = mean_absolute_percentage_error(y_actuals, in_sample_pred)

        rms = sqrt(mean_squared_error(data.ix[len(data)-5:,text], in_sample_pred))

        seriesname = "ARIMAX"

        lower, upper = list(), list()

        for i in range(len(confidence)):

            lower.append(confidence[i,0])
            upper.append(confidence[i,1])

        confidence = np.concatenate((np.array(lower),np.array(upper)))

        _original_ =  data.ix[:,text]

        output = graphing3(_original_, y_hat, confidence, rms, mape, seriesname, text, model_order, seasonal_order)

        args = {'form': form, 'output': output}

        return render(request, template_name , output)

    args = {'form': form, 'text': 'Pass the variable'}

    return render(request, 'polls/ARIMAX_output.html' , args)



def arimax_manual(request):

    template_name = 'polls/ARIMAX_man.html'

    if request.method == "POST":

        form = p_d_q(request.POST)
        #d = p_d_q(request.POST)
        #q = p_d_q(request.POST)
        if form.is_valid():

            p = form.cleaned_data['p']

            d = form.cleaned_data['d']

            q = form.cleaned_data['q']

            template_name = 'polls/ARIMAX_man.html'

            exog_var = (data_.drop(text, axis=1))#dependencies

            endog_var = data_.ix[:,text]#y_hat

            a = ARIMA(endog_var.astype('float64'), order = (p,d,q), exog = exog_var).fit()

            model_order = [p,d,q]
            seasonal_order = "Not applicable"

            modified = list()

            for i in range(0,12):

                temp = exog_var.ix[len(data_) - i: ]

            modified = a.forecast(steps = i, exog = temp.values)

            in_sample_pred = a.predict(start = len(exog_var) - 5 )

            y_actuals = data.ix[len(data)-5:, text].values

            y_hat, confidence = modified[0], modified[2]

            mape = mean_absolute_percentage_error(y_actuals, in_sample_pred)

            rms = sqrt(mean_squared_error(data.ix[len(data)-5:,text], in_sample_pred))

            seriesname = "ARIMAX"

            lower, upper = list(), list()

            for i in range(len(confidence)):

                lower.append(confidence[i,0])
                upper.append(confidence[i,1])

            confidence = np.concatenate((np.array(lower),np.array(upper)))

            _original_ =  data.ix[:,text]

            output = graphing3(_original_, y_hat, confidence, rms, mape, seriesname, text, model_order, seasonal_order)

            #args = {'form': form, 'output': output}

            return render(request, template_name , output)

    #args = {'form': form, 'text': 'Pass the variable'}

    return render(request, template_name , )


#SARIMAX graphing (isn't used yet)
def sarimax_graphing(_original_, modified, confidence, rmse, seriesname):

    tracey = list()
    k = 12
    for i in data.columns.values:

        pmet = list(data.ix[:,i])
        tracey.append(pmet)
        k = k + 12
        print i,k

    trace0 = list()
    trace1 = list()


    try:
         Index = dates

    except NameError:

        Index = len(data)

    for i in range(0, (len(data.columns.values))):

        trace0.append(go.Scatter(x = Index, y = _original_.ix[:, i], name = "Original",
                                     mode = "lines+markers"))

    for i in range(0, len(modified)):

        trace1.append(go.Scatter(x = [i for i in range(0, len(modified))], y = modified[i], name = "Forecast"),
                                     mode = "lines+markers")

        tempo = go.Data([i for i in trace0])

        X_name, Y_name = "Columns", "Values"

    layout= go.Layout(title="SARIMAX", xaxis={'title':X_name}, yaxis={'title':Y_name})

    figure=go.Figure(data=tempo,layout=layout)

    output = opy.plot(figure, auto_open=True, output_type='div')

    #output = {'output': output}

    return output

#For multiple variables of time series

def graphing3(original, modified, confidence, rmse, mape ,seriesname, text, model_order, seasonal_order):

    len_mod = len(modified)

    future_dates = (dating(dates))

    msbarChart = FusionCharts("msline", "ex1" , "950", "450", "chart-1", "json",
		# The chart data is passed as a string to the `dataSource` parameter.
      {
      "chart": {
        "caption": "Number of {} on the Y-axis and {} on X-axis" .format(text, data.index.names[0]),
         "subcaption": "Model Used {} , RMSE score: {} MAPE: {} model order: {} seasonal order: {}" .format(seriesname, round(rmse, 3), mape, model_order, seasonal_order),
          "linethickness": "2",
          "baseFontSize" : "14",
          "adjustDiv": "0",
          "showvalues": "0",
          "formatnumberscale": "0",
          "labeldisplay": "ROTATE",
          "enableiconmousecursors": "1",
          "axis":"linear",
          "dynamicaxis": "1",
          "slantlabels": "1",
          "divLineAlpha": "40",
          "anchoralpha": "0",
          "animation": "1",
          "legendborderalpha": "20",
          "drawCrossLine": "1",
          "crossLineColor": "#0d0d0d",
          "crossLineAlpha": "100",
          "tooltipGrayOutColor": "#80bfff",
          "yaxisminValue":min(original)-5,
          "yaxismaxValue":max(original)+5,
          "animation":"1",
          "theme": "fint",
      },
      "categories": [{
        "category": [[{'label' : i} for i in dates],
                     [{'label': i} for i in range(len_mod)]
                     ]
      },
    ],
      "dataset": [{
        "seriesname": "Original data",
        "data": [[{'value': i} for i in original],
                  [{'value': ' '} for x in range(len_mod)]]
      }, {
        "seriesname": "Forecast",
        "data": [[{'value': ' '} for i in range(len(dates))],
                 [{'value' : i} for i in modified]
                 ]
        },
    {
        "seriesname": "Lower_limit",
        "data": [[{'value': ' '} for i in range(len(dates))],
                 [{'value' : i} for i in confidence[:12]]]
        },
     {
        "seriesname": "Upper_limit",
        "data": [[{'value': ' '} for i in range(len(dates))],
                 [{'value' : i} for i in confidence[12:]]]
        }

     ]
    })

    output = {'output' : msbarChart.render()
              }

    return  output


#MA, ARMA and ARIMA
def graphing2(original, modified, confidence, rmse, mape, seriesname, _values_):

    len_mod = len(modified)

    future_dates = (dating(dates))

    msbarChart = FusionCharts("msline", "ex1" , "950", "450", "chart-1", "json",

      {
      "chart": {
        "caption": "Number of {} on the Y-axis and {} on X-axis" .format(data_clean_, data.index.names[0]),
         "subcaption": "Model Used {} , RMSE score: {} MAPE: {} p, d, q: {}" .format(seriesname, round(rmse, 3), mape, _values_),
          "linethickness": "2",
          "baseFontSize" : "14",
          "adjustDiv": "0",
          "showvalues": "0",
          "formatnumberscale": "0",
          "labeldisplay": "ROTATE",
          "enableiconmousecursors": "1",
          "axis":"linear",
          "dynamicaxis": "1",
          "slantlabels": "1",
          "divLineAlpha": "40",
          "anchoralpha": "0",
          "animation": "1",
          "drawCrossLine": "1",
          "crossLineColor": "#0d0d0d",
          "crossLineAlpha": "100",
          "tooltipGrayOutColor": "#80bfff",
          "yaxisminValue":min(original)-5,
          "yaxismaxValue":max(original)+5,
          "animation":"1",
          "theme": "fint",
      },
      "categories": [{
        "category": [[{'label' : i} for i in dates],
                     [{'label': i} for i in future_dates]
                     ]
      },
    ],
      "dataset": [{
        "seriesname": "Original data",
        "data": [[{'value': i} for i in original],
                  [{'value': ' '} for x in range(len_mod)]]
      }, {
        "seriesname": "Forecast",
        "data": [[{'value': ' '} for i in range(len(dates))],
                 [{'value' : i} for i in modified]]
        },
     {
        "seriesname": "Lower_limit",
        "data": [[{'value': ' '} for i in range(len(dates))],
                 [{'value' : i} for i in confidence[:17,0]]]
        },
     {
        "seriesname": "Upper_limit",
        "data": [[{'value': ' '} for i in range(len(dates))],
                 [{'value' : i} for i in confidence[:17,1]]]
        }

     ]
    })

    output = {'output' : msbarChart.render()}

    return  output


#Box plot
def Box_Whisker(request):
    no_use = dating(dates)
    template_name = 'polls/home.html'
    X_name, Y_name = str(), str()
    tempo = list()

    if data.index.name != "Numeric column" and len(data.columns)<2:

        tracey = list()
        k = 12
        for i in range(0, len(data) , 12):
            pmet = list(data.ix[i:k,data_clean_].values)
            tracey.append(pmet)
            k = k + 12
            print i,k

            if k > len(data):
                break

        trace0 = list()

        for i in range(0, (len(data)/12)):

            trace0.append(go.Box(y = tracey[i], name = yearing[i]))

        tempo = go.Data([i for i in trace0])

        X_name, Y_name = "Years", (data.ix[:,data_clean_]).name

    elif data.index.name == "Numeric column" or len(data.columns)>1 :

        tracey = list()
        k = 12
        for i in data.columns.values:

            pmet = list(data.ix[:,i])
            tracey.append(pmet)
            k = k + 12
            print i,k

        trace0 = list()

        for i in range(0, (len(data.columns.values))):

            trace0.append(go.Box(y = tracey[i], name = str(data.columns.values[i])))

        tempo = go.Data([i for i in trace0])

        X_name, Y_name = "Columns", "Values"

    layout= go.Layout(title="Box and Whiskers", xaxis={'title':X_name}, yaxis={'title':Y_name})

    figure=go.Figure(data=tempo,layout=layout)

    output = opy.plot(figure, auto_open=True, output_type='div')

    output = {"output": output}

    return render(request, template_name, output)


#linear regression
def Linear(request):

    template_name = 'polls/Graph_ouput_Linear.html'

    post_val = request.POST

    form = LinearForm(post_val)

    text = str()

    _data_column_names = (data.columns.values)

    l = [item.lower() for item in _data_column_names]

    a = tuple(_data_column_names)

    b = tuple(_data_column_names)

    disp_cols = tuple(zip(a,b))

    form = LinearForm(request.POST, choices = disp_cols)

    args = {'form': form, 'text': text}

    if request.method == "POST":

        if form.is_valid():

            text = form.cleaned_data['choice_field']

            l = [item.lower() for item in data.columns.values]

            if text.lower() in l:

                y = data.ix[:,str(text)]

                X = data.drop([text], axis = 1)

                residuals, y_hat, y_test = L_regression(X, y)

                new = pd.DataFrame()

                new = pd.DataFrame([y_test.values, y_hat, np.array(residuals)]).transpose()

                output = table_formatting(new)

                output = {'output': output}

                args = {'form': form, 'output': output}

                return render(request, template_name , output)

            else:

                y = data.ix[:, 0]

                X = data.drop([data.columns.values[0]], axis = 1)

                residuals, y_hat, y_test = L_regression(X, y)

                form = LinearForm(post_val)

                new = pd.DataFrame()

                new = pd.DataFrame([y_test.values, y_hat, np.array(residuals)]).transpose()

                output = table_formatting(new)

                args = {'form': form, 'text': output}

                return render(request, template_name , args)

    args = {'form': form, 'text': text}

    return render(request, template_name , args)


def table_formatting(new):

    tracey = list()

    for i in new.columns.values:

        print i
        pmet = list(new.ix[:,i])
        tracey.append(pmet)


    rmse = round(sqrt(mean_squared_error(y_test, y_hat)),2)

    mape = mean_absolute_percentage_error(y_test, y_hat)

    title_name = "Table RMSE: {} , MAPE: {} " .format(rmse, mape)

    trace = go.Table(
        header=dict(values=[i for i in ['Y_actuals', 'Predictions', 'Residuals']],
                    line = dict(color='#7D7F80'),
                    fill = dict(color='#a1c3d1'),
                    align = ['left'] * 5),
        cells=dict(values=[tracey[i] for i in range(0, len(new.columns.values))],
                   line = dict(color='#7D7F80'),
                   fill = dict(color='#EDFAFF'),
                   align = ['left'] * 5))

    tempo = go.Data([trace])

    if len(data.columns)>=3:
        width = 1000
        height = 1000

    else:

        width =1050
        height = 1000

    layout= go.Layout(autosize = True, width = width, height = height, title=title_name, xaxis={'title':'X'}, yaxis={'title':'Y'})

    figure=go.Figure(data=tempo,layout=layout)

    output = opy.plot(figure, auto_open=True, output_type='div')

    return output


#returns residuals, predicted out of sample and actuals of out of samples
def L_regression(X, y):

    #Will be need them for plotting the regression line
    global residuals, y_hat, y_test, lm_fit,  X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = int(len(X)*0.2))

    lm = linear_model.LinearRegression()

    lm_fit = lm.fit(X_train, y_train)

    y_hat = lm_fit.predict(X_test)

    residuals =  y_test - y_hat

    return residuals, y_hat, y_test


#plotting the regression line
def regression_plotly(request):

    template_name = 'polls/Graph_ouput_Linear.html'

    hypothesis = (X_test.dot(lm_fit.coef_))+ lm_fit.intercept_

    p1 = go.Scatter(x= y_test,
                y=y_hat,
                mode='markers',
                marker=dict(color='black')
               )

    p2 = go.Scatter(x= [hypothesis.min(), hypothesis.max()],
                y = [hypothesis.min(), hypothesis.max()],
                mode='lines',
                line=dict(color='blue', width=3)
                )

    tempo=go.Data([p1, p2])

    layout = go.Layout(title = "Regression Line", xaxis={'title':'X', 'ticks':'', 'showticklabels':True,
                              'zeroline':True},

                   yaxis={'title':'Y', 'ticks':'', 'showticklabels':True,
                              'zeroline':True},
                   showlegend=True, hovermode='closest')

    figure = go.Figure(data=tempo, layout=layout)

    output = opy.plot(figure, auto_open=False, output_type='div')

    output = {"output": output}

    return render(request, template_name, output)


#showing predicted values of linear
def predicted_values(request):

    template_name = 'polls/home.html'

    tracey = list()
    k = 12
    for i in backup_data.columns.values:

        pmet = list(backup_data.ix[:,i])
        tracey.append(pmet)
        k = k + 12
        print i,k


    trace = go.Table(
        header=dict(values=[i for i in backup_data.columns.values],
                    line = dict(color='#7D7F80'),
                    fill = dict(color='#a1c3d1'),
                    align = ['left'] * 5),
        cells=dict(values=[tracey[i] for i in range(0, len(backup_data.columns.values))],
                   line = dict(color='#7D7F80'),
                   fill = dict(color='#EDFAFF'),
                   align = ['left'] * 5))

    tempo = go.Data([trace])

    if len(data.columns)>=3:
        width = 1000
        height = 1000

    else:

        width =1050
        height = 1000

    layout= go.Layout(autosize = True, width = width, height = height, title="Predicted", xaxis={'title':'X'}, yaxis={'title':'Y'})

    figure=go.Figure(data=tempo,layout=layout)

    output = opy.plot(figure, auto_open=True, output_type='div')

    output = {'output': output}

    return render(request, template_name, output)


#For histogram residuals
def residual_distribution(request):

    template_name = 'polls/Graph_ouput_Linear.html'

    trace1 = go.Histogram(x = residuals)

    tempo=go.Data([trace1])

    layout=go.Layout(title="Histogram of Residuals", xaxis={'title':'RESIDS'}, yaxis={'title':'Values'})

    figure=go.Figure(data=tempo,layout=layout)

    output = opy.plot(figure, auto_open=False, output_type='div')

    output = {"output": output}

    return render(request, template_name, output)


def viewing_predicted_test_values(request):

    template_name = 'polls/Graph_ouput_Linear.html'

    resids = list(residuals.values)

    y_predicts = y_hat

    trace1 = go.Scatter(x = y_predicts, y= resids, marker={'color': 'blue', 'symbol': 14, 'size': "10"},
                        mode="markers",  name='1st Trace')#mode = "lines"

    tempo=go.Data([trace1])

    layout=go.Layout(title="Residuals vs Fitted values", xaxis={'title':'Fitted values'}, yaxis={'title':'Residuals'})

    figure=go.Figure(data=tempo,layout=layout)

    output = opy.plot(figure, auto_open=False, output_type='div')

    output = {"output": output}

    return render(request, template_name, output)


#Use this to select the column to forecast the values
def Select_column(request):

    post_val = request.POST

    form = LinearForm(post_val)

    text = pd.DataFrame()

    if request.method == "POST":

        if form.is_valid():

            text = form.cleaned_data['Column_name1']

            text = pd.DataFrame(data.ix[:,str(text)])

            return text

    return "Nothing returns"


def trail_graphical(request):

    template_name = 'polls/Components.html'

    output = lines_markers(data)

    output = {'output': output}

    return render(request, template_name, output)


def lines_markers(data, title_name = None):

    tracey = list()
    k = 12
    for i in data.columns.values:

        pmet = list(data.ix[:,i])
        tracey.append(pmet)
        k = k + 12
        print i,k

    trace0 = list()

    if data.index.name != "Numeric column" :

        Index = dates

    elif data.index.name == "Numeric column":

        Index = len(data)


    for i in range(0, (len(data.columns.values))):

        trace0.append(go.Scatter(x = Index, y = tracey[i], name = str(data.columns.values[i]),
                                     mode = "lines+markers"))

        tempo = go.Data([i for i in trace0])

        X_name, Y_name = "Columns", "Values"

    if title_name == None:

        if len(data.columns.values) < 2:

            title_name = str(data.columns.values[0])

        else:

            title_name = "The columns are plotted"

    layout= go.Layout(title=title_name, xaxis={'title':X_name}, yaxis={'title':Y_name})

    figure=go.Figure(data=tempo,layout=layout)

    output = opy.plot(figure, auto_open=True, output_type='div')

    del title_name

    return output



def table_plotly(request):

    template_name = 'polls/home.html'

    tracey = list()
    k = 12
    for i in backup_data.columns.values:

        pmet = list(backup_data.ix[:,i])
        tracey.append(pmet)
        k = k + 12
        print i,k


    trace = go.Table(
        header=dict(values=[i for i in backup_data.columns.values],
                    line = dict(color='#7D7F80'),
                    fill = dict(color='#a1c3d1'),
                    align = ['left'],
                    font = dict(family = 'Arial', size = 16)),
        cells=dict(values=[tracey[i] for i in range(0, len(backup_data.columns.values))],
                   line = dict(color='#7D7F80'),
                   fill = dict(color='#EDFAFF'),
                   align = ['left'],
                   font = dict(size = 14),
                   height = 30))

    tempo = go.Data([trace])

    if len(data.columns)>=3:
        width = 1500
        height = 1200

    else:

        width =1050
        height = 1200

    layout= go.Layout(autosize = True, width = width, height = height, title="Whole Dataset", xaxis={'title':'X'}, yaxis={'title':'Y'})

    figure=go.Figure(data=tempo,layout=layout)

    output = opy.plot(figure, auto_open=True, output_type='div')

    output = {'output': output}

    return render(request, template_name , output)


def summary_plotly(request):

    template_name = 'polls/home.html'

    data_ = data.describe()

    data_.insert(loc = 0, column = 'Row Names', value = data_.index)

    data_ = (data_.round(3))

    tracey = list()
    k = 12
    for i in data_.columns.values:
        print i
        pmet = list(data_.ix[:,i])
        tracey.append(pmet)
        k = k + 12
        print i,k


    trace = go.Table(
        header=dict(values=[i for i in data_.columns.values],
                    line = dict(color='#7D7F80'),
                    fill = dict(color='#a1c3d1'),
                    align = ['center'] * 5,
                    font = dict(size = 16)),
        cells=dict(values=[tracey[i] for i in range(0, len(data_.columns.values))],
                   line = dict(color='#7D7F80'),
                   fill = dict(color='#EDFAFF'),
                   align = ['center'] * 5,
                   font = dict(size = 14),
                   height = 30))

    tempo = go.Data([trace])

    if len(data_.columns)>=3:

        width = 1100
        height = 700

    else:


        width =1050
        height = 1000

    layout= go.Layout(autosize = True, width = width, height = height,
                       margin=go.Margin(
        l=50,
        r=50,
        b=150,
        t=100,
        pad=4

    ), title="Summary of the data", xaxis={'title':'X'}, yaxis={'title':'Y'})

    figure=go.Figure(data=tempo,layout = layout)

    output = opy.plot(figure, auto_open = True, output_type='div')

    output = {'output': output}

    return render(request, template_name, output)


#mean abolute percentage error
def mean_absolute_percentage_error(y_actuals, y_hat):

    mpe = np.mean(np.abs((y_actuals - y_hat) / y_actuals)) * 100
    #rounding it to two decimal points
    mpe = "{0:.2f}".format(round(mpe,2))

    return mpe
