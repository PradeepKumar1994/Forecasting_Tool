#ARIMAX
def sarimax_multi(request):
    
    template_name = 'polls/SARIMAX_output.html'
        
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
                    start_P=0, n_jobs=-1, d=1, D=0, trace=True,
                    error_action='ignore',  
                    suppress_warnings=True,  # don't want convergence warnings
                    stepwise=True, random=True, random_state=42,  # we can fit a random search (not exhaustive)
                    n_fits=25)
            
            a = ARIMA(endog_var.astype('float64'), order = model.order, exog = exog_var).fit()
            
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
            
            output = graphing3(_original_, y_hat, confidence, rms, mape, seriesname, text)
            
            args = {'form': form, 'output': output}
            
            return render(request, template_name , output)
        
    args = {'form': form, 'text': 'Pass the variable'}
    
    return render(request, 'polls/ARIMAX_output.html' , args)



def arimax_manual(request):
    
    template_name = 'polls/ARIMAX_man.html'
    
    if request.method == "POST":
        
        p = p_d_q(request.POST)
        d = p_d_q(request.POST)
        q = p_d_q(request.POST)
        
        if p.is_valid() and d.is_valid() and q.is_valid():
            
            p = p.cleaned_data['p']
            d = d.cleaned_data['d']
            q = q.cleaned_data['q']
    
            template_name = 'polls/ARIMAX_man.html'
    
            exog_var = (data_.drop(text, axis=1))#dependencies
            
            endog_var = data_.ix[:,text]#y_hat
            
            a = ARIMA(endog_var.astype('float64'), order = (p,d,q), exog = exog_var).fit()
            
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
            
            output = graphing3(_original_, y_hat, confidence, rms, mape, seriesname, text)
            
            #args = {'form': form, 'output': output}
            
            return render(request, template_name , output)
        
    #args = {'form': form, 'text': 'Pass the variable'}
    
    return render(request, template_name , )

