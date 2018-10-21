# -*- coding: utf-8 -*-
"""
Created on Wed Nov 01 11:30:09 2017

@author: pradeep kumar
"""

from django.conf.urls import url
from . import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
        
        url(r'^home/$', views.home , name = 'home'),
        
        url(r'^home/select_column/$', views.select_column, name = 'select_column'),
        
        url(r'^home/view_whole_plotly/$', views.table_plotly, name = 'view_whole_plotly'),
        
        url(r'^home/view_summary_plotly/$', views.summary_plotly, name = 'view_summary_plotly'),
        
        url(r'^home/moving-averages/$', views.moving_averages, name = 'moving_averages'),
        
        url(r'^home/exponential_smoothing/$', views.exponential_smoothing, 
            name = 'exponential_smoothing'),
            
        url(r'^home/exponential_smoothing_man/$', views.exponential_smoothing_manual, 
            name = 'exponential_smoothing_man'),
            
        url(r'^home/double_exponential_smoothing/$', views.double_exponential_smoothing, 
            name = 'double_exponential_smoothing'),
            
        url(r'^home/double_exponential_smoothing_man/$', views.double_exponential_smoothing_manual, 
            name = 'double_exponential_smoothing_man'),
            
        url(r'^home/triple_exponential_smoothing/$', views.triple_exponential_smoothing, 
            name = 'triple_exponential_smoothing'),
            
        url(r'^home/triple_exponential_smoothing_man/$', views.triple_exponential_smoothing_manual, 
            name = 'triple_exponential_smoothing_man'),
        
        url(r'^home/graphical_whole/$', views.trail_graphical, name = 'graphical_whole'),
        
        url(r'^home/ARMA/$', views.ARMA_model, name = 'ARMA'),
        
        url(r'^home/ARMA_auto/$', views.ARMA_model, name = 'ARMA_auto'),
        
        url(r'^home/ARMA_man/$', views.arma_manual, name = 'ARMA_man'),
        
        url(r'^home/ARIMA_auto/$', views.arima_auto, name = 'ARIMA_auto'),
        
        url(r'^home/ARIMA_man/$', views.arima_manual, name = 'ARIMA_man'),
        
        url(r'^home/ARIMAX_man/$', views.arimax_manual, name = 'ARIMAX_man'),
        
        url(r'^home/SARIMAX/$', views.sarimax_multi, name = 'SARIMAX'),
        
        url(r'^home/SARIMAX_man/$', views.sarimax_manual, name = 'SARIMAX_man'),
        
        url(r'^home/Stationarity/$', views.stationary_out, name = 'Stationarity'),
        
        url(r'^home/ARIMAX/$', views.arimax, name = 'ARIMAX'),
#       url(r'^home/ARIMA2/$', views.arima2, name = 'ARIMA2'),        
        url(r'^home/ARIMA2/$', views.arima_auto, name = 'ARIMA2'),
        
        url(r'^home/Naive/$', views.Naive_method, name = 'Naive'),
        
        url(r'^home/Box_Whisker/$', views.Box_Whisker, name = 'Box_Whisker'),
        
        url(r'^home/Resid_component/$', views.Resid_component, name = 'Resid_component'),
        
        url(r'^home/Trend_component/$', views.Trend_component, name = 'Trend_component'),
        
        url(r'^home/Seasonal_component/$', views.Seasonal_component, name = 'Seasonal_component'),
        
        url(r'^home/Stationarity_component/$', views.stationarity, name = 'Stationarity_component'),
        
        url(r'^home/Linear/$', views.Linear, name = 'Linear'),
        
        url(r'^home/residual_distribution/$', views.residual_distribution, name = 'residual_distribution'),
        
        url(r'^home/viewing_predicted_test_values/$', views.viewing_predicted_test_values, name = 'viewing_predicted_test_values'),
        
        url(r'^home/regression_line/$', views.regression_plotly, name = 'regression_line'),
        
        ]

if settings.DEBUG is True:
    urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)






