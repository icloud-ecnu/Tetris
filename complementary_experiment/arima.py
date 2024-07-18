import numpy as np
from time import time
from statsmodels.tsa.arima.model import  ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
import pmdarima 
from pmdarima.arima import auto_arima
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error,mean_absolute_percentage_error
import os
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

# 不确定pq参数
def ARIMA(cpuArray, end, forecastnum, pmax, qmax):
    cpu = np.array(cpuArray[0:int(end)]).astype(float) # 数据从int类型转为float
    bic_matrix = [] #bic矩阵
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try: #存在部分报错，所以用try来跳过报错。
                tmp.append(SARIMAX(cpu, order = (p,2,q)).fit(disp = False).bic) # ARIMA(p,2,q)模型
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
        
    bic_matrix = pd.DataFrame(bic_matrix) #从中可以找出最小值
    # print(bic_matrix) 
    p,q = bic_matrix.stack().astype(float).idxmin() #先用stack展平，然后用idxmin找出最小值位置。
    # print('BIC最小的p值和q值为：%s、%s' %(p,q))

    model = SARIMAX(cpu, order = (p,2,q)).fit(disp = False)
    yHat = model.forecast(forecastnum, alpha = 0.01) # 提高置信区间为99%
    print(yHat)
    return yHat

def getArimaModel(cpuArray, trainRatio, pmax=5, qmax=5):
    end = int(len(cpuArray) * trainRatio)
    # print(end)
    cpu = np.array(cpuArray[0:end]).astype(float) # 数据从int类型转为float, 使用70%的数据训练该容器的model
    bic_matrix = [] #bic矩阵
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try: #存在部分报错，所以用try来跳过报错。
                tmp.append(SARIMAX(cpu, order = (p,2,q)).fit(disp = False).bic) # ARIMA(p,2,q)模型
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
        
    bic_matrix = pd.DataFrame(bic_matrix) #从中可以找出最小值
    # print(bic_matrix) 
    # 检查 BIC 矩阵是否为空
    if bic_matrix.isnull().values.all():
        raise ValueError("BIC matrix is empty. Ensure that the input data is sufficient and valid.")
    
    # 检查 BIC 矩阵是否存在非空值
    if bic_matrix.stack().dropna().empty:
        raise ValueError("No valid BIC values found. Ensure that the input data is sufficient and valid.")
    
    try:
        p,q = bic_matrix.stack().astype(float).idxmin() #先用stack展平，然后用idxmin找出最小值位置。
        model = SARIMAX(cpu, order = (p,2,q)).fit(disp = False, maxiter=1000)
    except ValueError:
        model = auto_arima(cpu, trace=False, suppress_warnings=True)
        # 获取选择的模型参数
        p, d, q = model.order
        # 使用选择的参数构建并拟合ARIMA模型
        model = SARIMAX(cpu, order=(p, d, q)).fit(disp=False, maxiter=1000)
    # print('BIC最小的p值和q值为：%s、%s' %(p,q))

    return model

def forecastWithModel(model, start_idx, forecastnum):
    # 使用训练好的模型从指定的起始位置进行预测
    pred = model.get_prediction(start=start_idx, end=start_idx + forecastnum - 1)
    pred_conf = pred.conf_int(alpha=0.01)  
    yHat = pred.predicted_mean  # 获取预测值
    return yHat
# ARIMA模型预测
def arima_predict(cpu_list,end,W):
    cpuHist = np.array(cpu_list[0:int(end)])

    model =  pmdarima.arima.auto_arima(cpuHist)
    next_cpu = model.predict(W,alphas=0.01)
 
    # if (next_cpu < 0.001).all():
    #     next_cpu = np.zeros(W)

    return next_cpu

# 使用已有模型预测CPU和内存使用情况，并将结果存储在forecast_mem和forecast_cpu字典中
# end: 时间序列的结束索引，表示预测的起始点
def using_model(containerUtilDict,W,end,forecast_mem,forecast_cpu):
    for cid,util_data in containerUtilDict:
        cpu_list = [float(util[0]) for util in util_data if util[0] is not None]
        mem_list = [float(util[1]) for util in util_data if util[1] is not None]

        next_cpu,next_mem =  arima_predict(cpu_list,mem_list,end,W)
        # 不预测
        # next_cpu = instance.cpulist[end:end+W]
        # next_mem = instance.memlist[end:end+W]
        forecast_mem[cid] =next_mem
        forecast_cpu[cid] = next_cpu


# 使用模型预测并返回CPU和内存的预测值和实际值
def arimas(containerUtilDict,end,W):
    forecast_cpu = {}
    forecast_mem = {}
    using_model(containerUtilDict,W,int(end),forecast_mem,forecast_cpu)
    inc_cpu_pre = np.array([predicts for cid,predicts in forecast_cpu.items()])
    inc_mem_pre = np.array([predicts  for cid,predicts  in forecast_mem.items()])
    
    inc_actual = np.array([util_data[end:end+W] for cid,util_data in containerUtilDict.items()])
    inc_cpu_actual = np.array([util_data[0] for util_data in inc_actual])
    inc_mem_actual = np.array([util_data[1] for util_data in inc_actual])
    
    return inc_cpu_pre,inc_mem_pre,inc_cpu_actual,inc_mem_actual