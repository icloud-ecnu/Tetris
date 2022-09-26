import warnings 
import os                                 # do not disturbe mode
warnings.filterwarnings('ignore')
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Load packages
import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots

from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization

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
from  statsmodels.tsa.arima_model  import  ARIMA
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product                    # some useful functions
from tqdm import tqdm_notebook
from time import time

class ArimaForTrain():
    def __init__(self,filepath,trainlen,testlen):
        self.ver1_readData(filepath,trainlen,testlen)
        self.filepath = filepath
    
    
    def ver1_readData(self,filepath,trainlen,testlen):
        if trainlen == -1:
            data = pd.read_csv(filepath,header=None)
            self.data = data
            self.data.columns=['cpulist','memlist']
            df = data[:int(len(data)*0.7)]
            
            dftest = data[int(len(data)*0.7):]
        else :
            data = pd.read_csv(filepath,header=None,iterator=True)
            df = data.get_chunk(trainlen)
            dftest = data.get_chunk(testlen)
        
        # df.rename(columns={0:'cpulist'})
        # df.rename(columns={1:'memlist'})
        df.columns=['cpulist','memlist']
        dftest.columns= ['cpulist','memlist']
        self.df =df
        self.trainlen = len(df)
        #self.dftest = dftest
        self.actual = np.array(dftest['cpulist'])
        
        return df

    
    def auto(self,train,actual=None,criterion = 'aic'):
        model = pm.auto_arima(train, start_p=0, start_q=0,
                      information_criterion=criterion,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=5, max_q=5, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=False)

        #print(model.summary())

        # Forecast
        # n_periods = len(self.actual)
        # fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
        # mape = mean_absolute_percentage_error(fc,actual)
        # mape=format(mape, '.3f')
        return model
    
    
    def known_pqd(self,train,actual,p,d,q):
        model = SARIMAX(train,order=(p,d,q)).fit(disp=False)
        forecast = np.array(model.forecast(len(actual),alpha=0.01))
        mape = mean_absolute_percentage_error(forecast,self.actual)
        mape=format(mape, '.3f')
        # print(format(mape, '.3f'))
        return model,mape
    
    
    def param_product(self):
        ps = range(0, 5)
        d = range(0,3)
        qs = range(0, 5)
        # creating list with all the possible combinations of parameters
        parameters = product(ps, d,qs)
        return list(parameters)
   
    
    def optimizeSARIMA(self, parameters_list=None):
        """Return dataframe with parameters and corresponding AIC
            
            parameters_list - list with (p, q, P, Q) tuples
            d - integration order in ARIMA model
            D - seasonal integration order 
            s - length of season
        """
        
        results = []
        best_aic = float("inf")
        parameters_list = self.param_product()
        win,size = 24,6
        
        for param in tqdm_notebook(parameters_list):
            # we need try-except because on some combinations model fails to converge
            try:
                i = 0
                sumaic = 0
                idx =0
                
                while i < len(self.data["cpulist"])*0.7:
                    trainlist = self.data["cpulist"][i:i+win]
                    actual = self.data["cpulist"][i+win:i+win+size]
                    params = self.param_product()
                    model=sm.tsa.statespace.SARIMAX(trainlist, order=(param[0], param[1], param[2])).fit(disp=-1) 
                    aic = model.aic
                    sumaic += aic
                # saving best model, AIC and parameters
                    i += 6
                    idx +=1
                avgaic =sumaic / idx
                
                if avgaic < best_aic:
                    best_model = model
                    best_aic = aic
                    best_param = param
                results.append([param, avgaic])
            except:
                continue
            

        result_table = pd.DataFrame(results)
        result_table.columns = ['parameters', 'aic']
        # sorting in ascending order, the lower AIC is - the better
        result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
        p,d,q = result_table['parameters'][0][0],result_table['parameters'][0][1],result_table['parameters'][0][2]
        param = [p,d,q]
        # forecast = SARIMAX(trainlist,order=(p,d,q)).fit(disp=-1).forecast(len(actual),alpha=0.01) 
        # mape = mean_absolute_percentage_error(forecast,actual)
        # mape=format(mape, '.3f')
        # print('pqd = ',p,q,d,',',result_table['aic'][0],mape)
        return param
    
    
    def optimizeSARIMA_win(self, parameters_list,cpu):
        """Return dataframe with parameters and corresponding AIC
            
            parameters_list - list with (p, q, P, Q) tuples
            d - integration order in ARIMA model
            D - seasonal integration order 
            s - length of season
        """
        
        results = []
        best_aic = float("inf")

        for param in tqdm_notebook(parameters_list):
            # we need try-except because on some combinations model fails to converge
            try:
                model=sm.tsa.statespace.SARIMAX(cpu, order=(param[0], param[1], param[2])).fit(disp=-1) 
            except:
                continue
            aic = model.aic
            # saving best model, AIC and parameters
            if aic < best_aic:
                best_model = model
                best_aic = aic
                best_param = param
            results.append([param, model.aic])

        result_table = pd.DataFrame(results)
        result_table.columns = ['parameters', 'aic']
        # sorting in ascending order, the lower AIC is - the better
        result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
        
        return result_table
    
    
    def train_1(self,win,presize=6,lens=24):
        '''
        1. 从头开始训练 X
        2. 每次只训练win个 Y avg差
        3. 只训练一次 然后迭代更新
        '''
        
        trainlist = self.data["cpulist"][win-lens:win]
        actual = self.data["cpulist"][win:win+presize]
        params = self.param_product()
        tabel = self.optimizeSARIMA_win(params,trainlist)
        
        p,d,q = tabel['parameters'][0][0],tabel['parameters'][0][1],tabel['parameters'][0][2]
        param = [p,d,q]
        forecast = SARIMAX(trainlist,order=(p,d,q)).fit(disp=-1).forecast(len(actual),alpha=0.01) 
        mape = mean_absolute_percentage_error(forecast,actual)
        mape=format(mape, '.3f')
        
        print('pqd = ',p,q,d,',',tabel['aic'][0],mape)
        return param,mape

    
    def reduce0(self,forecast,actual):
        index = np.array(np.where(actual==0))
        new_ac = np.delete(actual,index)
        new_fore = np.delete(forecast,index)
        
        return new_fore,new_ac

    
    def test(self,params:dict,win=30,testsize=6):
        size = int(len(self.data["cpulist"])*0.7)
        i = size+testsize*2
        idx = 0
        sum = 0.0
        autospend = 0
        alltimestart = time()
        param = list(params.keys())
        
        while i < len(self.data["cpulist"]):
            print(f'TESE {self.filepath} in i = {i}')
            data = np.array(self.data["cpulist"][i-win-testsize:i-testsize])
            actual = np.array(self.data["cpulist"][i-testsize:i])
            #params = self.param_product()
            #tabel = self.optimizeSARIMA(params,data)
            # p,d,q = tabel['parameters'][0][0],tabel['parameters'][0][1],tabel['parameters'][0][2]
            # print(p,d,q)
            i += 1
            
            try:
                model = SARIMAX(data,order=(param[0][0], param[0][0], param[0][0])).fit(disp=-1)
                forecast=model.forecast(len(actual),alpha=0.01) 
            except:
                start = time()
                try:
                    model = SARIMAX(data,order=(param[0][1], param[0][1], param[0][1])).fit(disp=-1)
                    forecast=model.forecast(len(actual),alpha=0.01)
                except:
                    modelauto = self.auto(data)
                    forecast = modelauto.predict(n_periods=testsize)
                
                end = time()
                autospend += end-start
                #print('wrong',format(model.aic,'.3f'))
            
            print(type(forecast),type(actual))
            print(forecast,actual)
            forecast,actual = self.reduce0(forecast,actual)
            
            if actual.shape[0] != 0:
                mape = mean_absolute_percentage_error(forecast,actual)
            else:
                mape = 0
            mape=format(mape, '.3f')
            sum += float(mape)
            print(mape,format(model.aic,'.3f'))
            idx+=1
        
        avg = sum/idx
        allendtime = time()
        alltime = alltimestart -allendtime
        zhanbi =100* autospend / alltime
        
        with open('/hdd/lsh/Scheduler/test.txt','a') as f:
            f.write(f'{self.filepath}:  avgmape =  {avg}  auto model spend = {autospend}  alltime = {alltime}\
            zhanbi = {zhanbi}% , \n')
        f.close()

if __name__ == "__main__":
    filepath = '/hdd/jbinin/AlibabaData/target/'
    files = os.listdir(filepath)
    presize = 6
    trianlens = 24
    
    for idx,file in enumerate(files):
        if idx < 6:
            continue
        filename = os.path.join(filepath, file)
        print(f'train {filename}')
        
        ver2 = ArimaForTrain(filename,-1,20)
        sum = 0
        size = 0
        params = {}
        mapes = {}
        
        for win in range(24,ver2.trainlen,3):
            print('win =',win,'len =',ver2.trainlen)
            p,mape = ver2.train_1(win,presize,trianlens)
            p = tuple(p)
            
            if p in params:
                params[p] =  params[p]+1
                mapes[p] += float(mape)
            else:
                params[p] =  1
                mapes[p] =float(mape)
            size=size+1
        
        for k,v in  mapes.items():
            mapes[k] = float(v) / float(params[k])

        with open('/hdd/lsh/Scheduler/metric.txt','a') as f:
            f.write(f'{filename},{params},{mape} \n')
        f.close()
        params = {k:v for k,v in sorted(params.items(),key=lambda x:x[1])}
        ver2.test(params)