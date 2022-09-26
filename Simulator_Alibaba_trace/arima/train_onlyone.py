import warnings 
import os
from pyparsing import alphas                                 # do not disturbe mode
warnings.filterwarnings('ignore')
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Load packages
import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots

from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization
import statsmodels
import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product                    # some useful functions
from tqdm import tqdm_notebook

# Importing everything from forecasting quality metrics
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error,mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller
from time import time
from  statsmodels.tsa.arima.model import  ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
import pmdarima 
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product                    # some useful functions
from tqdm import tqdm_notebook
from time import time
import json


# def __getnewargs__(self):
#     return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
# SARIMAX.__getnewargs__ = __getnewargs__


def optimizeSARIMA(cpu):
        """Return dataframe with parameters and corresponding AIC
            
            parameters_list - list with (p, q, P, Q) tuples
            d - integration order in ARIMA model
            D - seasonal integration order 
            s - length of season
        """
        ps = range(0, 5)
        d=range(0,3)
        qs = range(0, 5)
        # creating list with all the possible combinations of parameters
        parameters_list = product(ps, d,qs)
        # results = []
        best_aic = float("inf")
       
        for param in tqdm_notebook(parameters_list):
            # we need try-except because on some combinations model fails to converge
            try:
                # model=sm.tsa.statespace.SARIMAX(cpu, order=(param[0], param[1], param[2])).fit(disp=-1) 
                model = sm.tsa.arima.ARIMA(cpu, order=(param[0], param[1], param[2])).fit() 
                aic = model.aic
                # print(aic)
            except:
                continue
           
            # saving best model, AIC and parameters
            if aic < best_aic:
                best_model = model
                # best_aic = aic
                best_param = param
            # results.append([param, model.aic])
        print(best_param)
        
        return best_model ,best_param

    
def reduce0(forecast,actual):
    index = np.array(np.where(actual==0))
    new_ac = np.delete(actual,index)
    new_fore = np.delete(forecast,index)
    
    return new_fore,new_ac


def train():
    filepath = '/hdd/jbinin/AlibabaData/target/'
    files = os.listdir(filepath)
  
    for idx,file in enumerate(files):
        filename = os.path.join(filepath, file)
        start = time()
        savemodel(filename)
        end = time()
        all =end-start
        print(f'spending {all}s')   
    pass


def demo1():
    filename = '/hdd/jbinin/AlibabaData/target/instanceid_12199.csv'
    data = pd.read_csv(filename,header=None)
    data.columns=['cpulist','memlist']
    lens = int(len(data)*0.7)
    df = np.array(data[:lens]['cpulist'])
    
    actual = np.array(data[lens:]['cpulist'])
    model , param= optimizeSARIMA(df)
    
    print(param)
    forecast=model.forecast(len(actual),alpha=0.01)
    print(forecast)
    forecast,actual = reduce0(forecast,actual)
    
    if actual.shape[0] != 0:
        mape = mean_absolute_percentage_error(forecast,actual)
    else:
        mape = 0
    # with open('/hdd/lsh/Scheduler/arima/pqd.txt','a') as f:
    #     f.write(f'{filename},{params},{mape} \n')
    # f.close()
    print(f'train {filename}, mape = {mape} ')
    
    model=sm.tsa.arima.ARIMA(df, order=(param[0],param[1],param[2])).fit()
    forecast=model.forecast(len(actual),alpha=0.01)
    print(forecast)
    
    return model,param


def is_number(x):
    try:
        int(x)
    except:
        return False
    
    return True


def pqd(model):
    pqd = model.summary().tables[0].data[1][1]
    pqd = pqd[pqd.find('(')+1:pqd.find(')')]
    print(pqd)
    i,j=0,0
    l = []
    
    while i<len(pqd) :
        j=i+1
        sums = 0
        flag = False
        
        while j<=len(pqd):
            if is_number(pqd[i:j]):
                sums= sums*10+int(pqd[i:j])
                flag = True
            else:
                break
            j = j+1
        
        if flag:
            l.append(sums)
        i =i+1
    
    try:
        t = str(l[0])+str(l[1])+str(l[2])
    except:
        t='000'
    
    #print('t:',t)
    return t


def adddict(ma:dict,k,v):
    if k not in ma:
        ma[k] = [v]
    else:
        ma[k].append(v)

        
def jsondump(path,dic):
    with open(path,'w') as f:
        json.dump(dic,f)
    f.close()


def auto_wrong_test():
    file = '/hdd/jbinin/AlibabaData/target/instanceid_30287.csv'
    data = pd.read_csv(file,header=None)
    data.columns=['cpulist','memlist']
    lens = int(len(data)*0.7)
    df = np.array(data[:lens]['cpulist'])
    dftest = np.array(data[lens:lens+10]['cpulist'])
    df_mem = np.array(data[:lens]['memlist'])
    model = pmdarima.arima.auto_arima(df,start_p=0,start_q=0,alpha=0.01,\
                                      trace=False,stepwise=False,suppress_warnings=True,error_action='ignore')
    pqd = model.summary().tables[0].data[1][1]
    print(pqd)


def save_auto():
    model_inc_cpu = {}
    inc_model_cpu = {}
    model_inc_mem = {}
    inc_model_mem = {}
    
    filepath = '/hdd/jbinin/AlibabaData/target/'
    files = os.listdir(filepath)
    path_cpu = '/hdd/lsh/Scheduler/arima/models_cpu/'
    path_mem = '/hdd/lsh/Scheduler/arima/models_mem/'
    i = 0
    
    for idx,filename in enumerate(files):
        i+=1
        ids = filename[filename.rfind('_')+1:filename.rfind('.')]
        file = os.path.join(filepath,filename)
        # test
        # file = '/hdd/jbinin/AlibabaData/target/instanceid_1.csv'
        print('i=',i,'file=',file)
        
        data = pd.read_csv(file,header=None)
        data.columns=['cpulist','memlist']
        lens = int(len(data)*0.7)
        df = np.array(data[:lens]['cpulist'])
        dftest = np.array(data[lens:lens+10]['cpulist'])
        df_mem = np.array(data[:lens]['memlist'])
        
        model_cpu = pmdarima.arima.auto_arima(df,start_p=0,start_q=0,alpha=0.01,\
                                              trace=False,stepwise=False,suppress_warnings=True,error_action='ignore')
        # print('i=',i,'mem')
        model_mem =  pmdarima.arima.auto_arima(df_mem,start_p=0,start_q=0,alpha=0.01,\
                                               trace=False,stepwise=False,suppress_warnings=True,error_action='ignore')
        
        t_cpu = pqd(model_cpu)
        t_mem = pqd(model_mem)
        print('cpu:',t_cpu)
        print('mem',t_mem)
        
        newfilename_cpu = t_cpu+'.pkl'
        newfilename_mem = t_mem+'.pkl'
       
        cpupath = os.path.join(path_cpu,newfilename_cpu)
        mempath = os.path.join(path_mem,newfilename_mem)
        
        if not os.path.exists(cpupath):
            model_cpu = sm.tsa.arima.ARIMA(data['cpulist'], order=(int(t_cpu[0]),int(t_cpu[1]),int(t_cpu[2]))).fit()
            model_cpu.save(cpupath)
        
        if not os.path.exists(mempath):
            model_mem = sm.tsa.arima.ARIMA(data['memlist'], order=(int(t_mem[0]),int(t_mem[1]),int(t_mem[2]))).fit() 
            model_mem.save(mempath)
        
        adddict(model_inc_cpu,t_cpu,ids)
        inc_model_cpu[int(ids)]= t_cpu
        adddict(model_inc_mem,t_mem,ids)
        inc_model_mem[int(ids)]= t_mem
        
        jsondump('/hdd/lsh/Scheduler/arima/json/model_inc_cpu.json',model_inc_cpu)
        jsondump('/hdd/lsh/Scheduler/arima/json/inc_model_cpu.json',inc_model_cpu)
        jsondump('/hdd/lsh/Scheduler/arima/json/model_inc_mem.json',model_inc_mem)
        jsondump('/hdd/lsh/Scheduler/arima/json/inc_model_mem.json',inc_model_mem)

    
def demo_auto(filename):
    # filename = '/hdd/jbinin/AlibabaData/target/instanceid_8215.csv'
    data = pd.read_csv(filename,header=None)
    data.columns=['cpulist','memlist']
    lens = int(len(data)*0.7)
    df = np.array(data[:lens]['cpulist'])
    
    actual = np.array(data[lens:lens+10]['cpulist'])
    start = time()
    model,param= optimizeSARIMA(df)
    end = time()
    model_auto = pmdarima.arima.auto_arima(df,start_p=0,start_q=0,alpha=0.01,\
                                           trace=True,stepwise=False,suppress_warnings=True,error_action='ignore')
    # model_auto.update(actual)
    autoend = time()
    modeltime = end-start
    autotime= autoend-end
    print(df)
    print(f'model time = {modeltime} auto time = {autotime}')
    
    forecast=model.forecast(len(actual),alpha=0.01)
    n_diff = pmdarima.arima.ndiffs(df, test='adf', max_d=5)
    print(f'n_diff = {n_diff}')
    
    forecast_auto=np.array( model_auto.predict(n_periods=len(actual)))
    
    smape = np.mean(2*np.abs(forecast-actual)/(forecast+actual))
    smapeauto = np.mean(2*np.abs(forecast_auto-actual)/(forecast_auto+actual))
    print('model:',forecast)
    print('auto model:',forecast_auto)
    print('actual:',actual)
    
    forecast_auto,actual = reduce0(forecast_auto,actual)
    forecast,actual = reduce0(forecast,actual)
    
    if actual.shape[0] != 0:
        mape = mean_absolute_percentage_error(forecast,actual)
        mapeauto = mean_absolute_percentage_error(forecast_auto,actual)
    else:
        mape = 0
        mapeauto = 0
    print(f' pre = {mape} auto_pre = {mapeauto} ,smape_pre = {smape} ,smape_auto={smapeauto}')
    
    return modeltime,mape,autotime,mapeauto


def demo_pqd(filename):
    # filename = '/hdd/jbinin/AlibabaData/target/instanceid_8215.csv'
    data = pd.read_csv(filename,header=None)
    data.columns=['cpulist','memlist']
    lens = int(len(data)*0.7)
    df = np.array(data[:lens]['cpulist'])
    
    actual = np.array(data[lens:lens+10]['cpulist'])
    model=sm.tsa.statespace.SARIMAX(df, order=(2,1,2)).fit(disp=-1)
    forecast=model.forecast(len(actual),alpha=0.01)
    
    forecast,actual = reduce0(forecast,actual)
    
    if actual.shape[0] != 0:
        mape = mean_absolute_percentage_error(forecast,actual)
        
    else:
        mape = 0
    print(mape)
    
    
def savemodel(filename):
    # filename = '/hdd/jbinin/AlibabaData/target/instanceid_8215.csv'
    ids = filename[filename.rfind('_')+1:filename.rfind('.')]+'.pkl'
    path = '/hdd/lsh/Scheduler/arima/model/'
    savepath = os.path.join(path,ids)
    
    if os.path.exists(savepath):
        loadmodel = ARIMAResults.load(savepath)
        pqd = loadmodel.summary()
        l = pqd.find('(')+1
        r = pqd.find(')')
        p = pqd[l:l+1]
        d= pqd[l+3:l+4]
        q= pqd[l+6:l+7]
        
        return 
    
    data = pd.read_csv(filename,header=None)
    data.columns=['cpulist','memlist']
    lens = int(len(data)*0.7)
    df = np.array(data[:lens]['cpulist'])
    model = optimizeSARIMA(df)
    
    model.save(savepath)
    # forecast1=np.array( model.forecast(len(actual),alpha=0.01))
    # loadmodel = ARIMAResults.load(savepath)
    # newdatat = np.random.rand(1000000)
    # # actual = np.array(data[lens:lens+20]['cpulist'])
    # # forecast= np.array( loadmodel.forecast(len(actual),alpha=0.01))
    # start = time()
    # loadmodel = loadmodel.apply(df,refit=True)
    
    # forecast2= np.array( loadmodel.forecast(len(actual),alpha=0.01))
    # end = time()
    # print(end -start)
    # print (np.mean(np.abs((forecast2-actual))/actual))
    # print (np.mean(np.abs((forecast-actual))/actual))


def remodel(filepath):
    files = os.listdir(filepath)
    models = {}
    inc_model = {}
    
    for filename in files:
        path = '/hdd/lsh/Scheduler/arima/models/'
        loadpath = os.path.join(filepath,filename)
        loadmodel = ARIMAResults.load(loadpath)
        ids = filename[:filename.find('.')]
        
        pqd = loadmodel.summary().tables[0].data[1][1]
        l = pqd.find('(')+1
        # r = pqd.find(')')
        p = pqd[l:l+1]
        d = pqd[l+3:l+4]
        q = pqd[l+6:l+7]
        newfilename = str(p)+str(q)+str(d)+'.pkl'
        newloalpath = os.path.join(path,newfilename)
        
        if not os.path.exists(newloalpath):
            loadmodel.save(newloalpath)
        print(filename,newfilename)
        t = str(p)+str(q)+str(d)
        
        if t not in inc_model:
            
            inc_model[t] = []
        else:
            inc_model[t].append(ids)
        models[int(ids)]= t
    
    with open('/hdd/lsh/Scheduler/arima/model_json.json','w') as f:
        json.dump(models,f)
    
    with open('/hdd/lsh/Scheduler/arima/inc_model.json','w') as c:
        json.dump(inc_model,c)
    f.close()
    c.close()    


def test(datafile,modelfile):
    data = pd.read_csv(datafile,header=None)
    data.columns=['cpulist','memlist']
    lens = int(len(data)*0.7)
    df = np.array(data[:lens]['cpulist'])*100
    print(df)
    
    actual = np.array(data[lens:lens+10]['cpulist'])*100
    model = ARIMAResults.load(modelfile).apply(df,refit=True)
    model_auto = pmdarima.arima.AutoARIMA(start_p=0,start_q=0, max_d =3, trace=True,stepwise=False,\
                                          suppress_warnings=True,error_action='ignore').fit(df)
    model_auto_arima = pmdarima.arima.auto_arima(df,start_p=0,start_q=0,alpha=0.01, trace=True,stepwise=False,\
                                                 suppress_warnings=True,error_action='ignore')

    forecast = model_auto.predict(len(actual),alphas=0.01)
    mape=np.mean(np.abs(forecast-actual)/(actual))
    smape = np.mean(2*np.abs(forecast-actual)/(forecast+actual))
    rmse =  np.mean((forecast-actual)**2)
    print(f'AutoARIMA,mape={mape}smape={smape} rmse={rmse}')
    
    forecast = model.forecast(len(actual),alphas=0.01)
    mape=np.mean(np.abs(forecast-actual)/(actual))
    smape = np.mean(2*np.abs(forecast-actual)/(forecast+actual))
    rmse =  np.mean((forecast-actual)**2)
    print(f'pqd=101 :mape={mape}smape={smape} rmse={rmse}')
    
    forecast = model_auto_arima.predict(len(actual),alphas=0.01)
    mape=np.mean(np.abs(forecast-actual)/(actual))
    smape = np.mean(2*np.abs(forecast-actual)/(forecast+actual))
    rmse =  np.mean((forecast-actual)**2)
    print(f'auto_arima: mape={mape}smape={smape} rmse={rmse}')
    

if __name__ == "__main__":
    filename = '/hdd/jbinin/AlibabaData/target/instanceid_19.csv'
    modelfilename = '/hdd/lsh/Scheduler/arima/models_cpu/101.pkl'
    # remodel(modelfilename)
    # test('/hdd/lsh/Scheduler/arima/models/432.pkl')
    # demo1()
    # demo_auto(filename)
    # save_auto()
    # test(filename,modelfilename)
    # filepath = '/hdd/jbinin/AlibabaData/target/'
    # files = list(os.listdir(filepath))
    # testid=files.index('instanceid_1.csv')
    # print(files.index('instanceid_1.csv'))
    # file = os.path.join(filepath,files[testid])
    # data = pd.read_csv(file,header=None)
    # data.columns=['cpulist','memlist']
    # lens = int(len(data)*0.7)
    # df = np.array(data[:lens]['cpulist'])
    # actual = np.array(data[lens:lens+10]['cpulist'])
    # df_mem = np.array(data[:lens]['memlist'])
    # model= pmdarima.arima.auto_arima(df,start_p=0,start_q=0,alpha=0.01, trace=True,stepwise=False,\
    #                                  suppress_warnings=True,error_action='ignore')
    # forecast = model.predict(len(actual),alphas=0.01)
    # mape=np.mean(np.abs(forecast-actual)/(actual))
    # smape = np.mean(2*np.abs(forecast-actual)/(forecast+actual))
    # rmse =  np.mean((forecast-actual)**2)
    # print(f'auto_arima: mape={mape}smape={smape} rmse={rmse}')
   
    # with open('/hdd/lsh/Scheduler/arima/json/inc_model_cpu.json','r') as f:
    #     ma = json.load(f)
    # x = len(ma.keys())
    # print(x)
    save_auto()
