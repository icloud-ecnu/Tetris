import numpy as np
from time import time
from statsmodels.tsa.arima.model import  ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
import pmdarima 
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error,mean_absolute_percentage_error
import os
import statsmodels.api as sm
from cluster import Cluster


def build_newmodel(cpulist,model_cpu,path_cpu):
    t_cpu = pqd(model_cpu)
    newfilename_cpu = t_cpu+'.pkl'
    cpupath = os.path.join(path_cpu,newfilename_cpu)
    
    if not os.path.exists(cpupath):
        model_cpu_new = sm.tsa.arima.ARIMA(cpulist, order=(int(t_cpu[0]),int(t_cpu[1]),int(t_cpu[2]))).fit()
        model_cpu_new.save(cpupath)

        
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
    # print('t:',t)
    return t


def arima_speed_test(cluster:Cluster,inc_id,instance,end,W):
    cpuHist = np.array(instance.cpulist[0:int(end+1)])
    memHist = np.array(instance.memlist[0:int(end+1)])
    cpu_wrong ,mem_wrong= 0,0
    inc_id_old = str(cluster.inc_ids[inc_id])
    
    # 获取当前container的cpu和mem模型
    try:
        
        model_cpu = cluster.inccpu_model[inc_id]
        next_cpu = model_cpu.forecast(W,alphas=0.01)
        flag = True
        cluster.inccpu_model[inc_id] = model_cpu.append(np.array([instance.cpulist[int(end):int(end+1)]]))
    except:
        # print(f'\t\tcontainer inc_id = {inc_id} 的 cpu model不行')
        cpu_wrong +=1
        model =  pmdarima.arima.auto_arima(cpuHist)
        next_cpu = model.predict(W,alphas=0.01)
        # build_newmodel(cpuHist,model,'/hdd/lsh/Scheduler/arima/models_cpu')
        
    try:            
        model_mem =  cluster.incmem_model[inc_id]
        next_mem = model_mem.forecast(W,alphas=0.01)           
        cluster.incmem_model[inc_id] = model_mem.append(np.array([instance.cpulist[int(end):int(end+1)]]))            
    except:
        # print(f'\t\tcontainer inc_id = {inc_id} 的 mem model不行')
        mem_wrong += 1
        model =  pmdarima.arima.auto_arima(memHist)
        next_mem= model.predict(W,alphas=0.01)
        # build_newmodel(memHist,'/hdd/lsh/Scheduler/arima/models_mem')
    # print(f'next cpu = {next_cpu.tolist()} next mem = {next_mem.tolist()}')  
    
    if (next_cpu < 0.001).all():
        next_cpu = np.zeros(W)
    if (next_mem < 0.001).all():
        next_mem = np.zeros(W)
    if cpu_wrong != 0 or mem_wrong!=0:
        print(f'\t\tat instance {inc_id} wrong : cpuwrong = {cpu_wrong} memwrong= {mem_wrong}')
    
    return next_cpu,next_mem


def using_model(cluster:Cluster,W,end,forecast_mem,forecast_cpu):
    startPredict = time()
    
    for inc_id,instance in cluster.instances.items():
        s = time()
        # next_cpu,next_mem =  arima_speed_test(cluster,inc_id,instance,end,W)
        # print(f'\t\tat instance {inc_id} spending {time()-s}s')
        # 不预测
        next_cpu = instance.cpulist[end:end+W]
        next_mem = instance.memlist[end:end+W]
        forecast_mem[inc_id] =next_mem
        forecast_cpu[inc_id] = next_cpu

    after =  time()
    # print(f'at {end} ariam predict: {after-startPredict}s')

    
def reduce0(forecast,actual):
    index = np.array(np.where(actual==0))
    # print(f'in reduce: index={index} actual={actual}')
    new_ac = np.delete(actual,index)
    new_fore = np.delete(forecast,index)
    
    return new_fore,new_ac


def arimas(cluster,end,W):
    forecast_cpu = {}
    forecast_mem = {}
    using_model(cluster,W,int(end),forecast_mem,forecast_cpu)
    inc_cpu_pre = np.array([predicts for inc_id,predicts in forecast_cpu.items()])
    inc_mem_pre = np.array([predicts  for inc_id,predicts  in forecast_mem.items()])
    inc_cpu_actual = np.array([inc.cpulist[end:end+W] for inc_id,inc in cluster.instances.items()])
    inc_mem_actual = np.array([inc.memlist[end:end+W] for inc_id,inc in cluster.instances.items()])
    
    return inc_cpu_pre,inc_mem_pre,inc_cpu_actual,inc_mem_actual