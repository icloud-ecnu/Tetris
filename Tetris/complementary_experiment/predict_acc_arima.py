import pandas as pd
import numpy as np
import ijson
import json
import os
from time import time
from arima import getArimaModel, forecastWithModel
import psutil
import warnings
warnings.simplefilter('ignore', category=Warning)


def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"RSS: {memory_info.rss / 1024 ** 2:.2f} MB")

nonperiodic_cpu=set(pd.read_csv('./nonperiodic_cpu.csv')['id'].values.tolist())
periodic_cpu=set(pd.read_csv('./periodic_cpu.csv')['id'].values.tolist())
nonperiodic_mem=set(pd.read_csv('./nonperiodic_mem.csv')['id'].values.tolist())
periodic_mem=set(pd.read_csv('./periodic_mem.csv')['id'].values.tolist())

periodic_inc=periodic_cpu | periodic_mem
nonperiodic_inc=nonperiodic_cpu | nonperiodic_mem
# print(f'number: periodic cpu {len(periodic_cpu)}, periodic mem {len(periodic_mem)}, nonperiodic cpu {len(nonperiodic_cpu)}, nonperiodic mem {len(nonperiodic_mem)}')

def MAPE(predict_value,real_value):
    real_value = np.array(real_value)
    predict_value = np.array(predict_value)
    # 过滤掉real_value中的零值
    mask = real_value != 0
    if np.sum(mask) == 0:
        return np.nan  # 如果过滤后没有有效数据，返回NaN
    real_value = real_value[mask]
    predict_value = predict_value[mask]
    return 100 * np.mean(np.abs((predict_value - real_value) / real_value))

cpuMapeList=list()
memMapeList=list()
np_cpuMapeList=list()
np_memMapeList=list()
w=6
count=0

# 获取每类的util的时间序列
with open('/mnt/d/school/tetris/containerUtilDict.txt', 'r') as file:
    parser=ijson.kvitems(file, '')
    print('get parser')
    for cid, util_data in parser:
        # if count<=1832:
        #     count+=1
        #     continue
        print(f'count is {count}')
        if count==4000:
            break
        cpu_list = [float(util[0]) for util in util_data if util[0] is not None]
        mem_list = [float(util[1]) for util in util_data if util[1] is not None]
        # 清理数据，去除无效值
        cpulist = [x for x in cpu_list if not np.isnan(x) and np.isfinite(x)]
        memlist = [x for x in mem_list if not np.isnan(x) and np.isfinite(x)]
        # print(cpulist)
        # print(memlist)
        cpu_mapelist=list()
        mem_mapelist=list()
        np_cpu_mapelist=list()
        np_mem_mapelist=list()

        cpu_trainRatio=1
        # cpu_startIdx=0
        cpu_startIdx = int(len(cpulist) * cpu_trainRatio)
        if cpu_startIdx==len(cpulist):
           cpu_startIdx=int(cpu_startIdx*0.7)

        mem_trainRatio=0.7
        # mem_startIdx=0
        mem_startIdx = int(len(memlist) * mem_trainRatio)
        if mem_startIdx==len(memlist):
            mem_startIdx=int(mem_startIdx*0.7)

        if cid in periodic_cpu:
            try:
                print("Before loading model:")
                print_memory_usage()
                cpu_arima_model = getArimaModel(cpulist, cpu_trainRatio, 5, 5)
                print("After loading model:")
                print_memory_usage()
            except Exception:
                continue
            hour=0
            start_time=time()
            for startidx in range(cpu_startIdx, len(cpulist)-w+1):
                if hour==24:
                    break
                try:
                    next_cpu = forecastWithModel(cpu_arima_model, startidx, w)
                    # print(f'next cpu is {next_cpu}')
                except Exception as e:
                    print(f"ARIMA 预测时出现错误：{str(e)}")
                    continue
                
                real_cpu = cpulist[startidx:startidx+w]
                if len(next_cpu) != len(real_cpu):
                    continue
                cpu_mape = MAPE(next_cpu, real_cpu)
                if not np.isnan(cpu_mape):
                    cpu_mapelist.append(cpu_mape)
                hour+=1
            end_time=time()
            print(f'{cid}: 周期性cpu，处理时间{end_time-start_time}, 平均处理时间{(end_time-start_time)/hour}')
        if cid in periodic_mem:
            try:
                print("Before loading model:")
                print_memory_usage()
                mem_arima_model = getArimaModel(memlist, mem_trainRatio, 5, 5)
                print("After loading model:")
                print_memory_usage()
            except Exception:
                continue
            hour=0
            start_time=time()
            for startidx in range(mem_startIdx, len(memlist)-w+1):
                if hour==24:
                    break
                try:
                    next_mem = forecastWithModel(mem_arima_model, startidx, w)
                    
                except Exception as e:
                    print(f"ARIMA 预测时出现错误：{str(e)}")
                    continue
                
                real_mem = memlist[startidx:startidx+w]
                print(f'{hour}:real_mem is {real_mem}, next mem is {next_mem}')
                if len(next_mem) != len(real_mem):
                    continue
                mem_mape = MAPE(next_mem, real_mem)
                print(f'{hour}:mem_mape is {mem_mape}')
                if not np.isnan(mem_mape):
                    mem_mapelist.append(mem_mape)
                hour+=1
            end_time=time()
            print(f'{cid}: 周期性mem，处理时间{end_time-start_time}, 平均处理时间{(end_time-start_time)/hour}')
        if cid in nonperiodic_cpu:
            try:
                print("Before loading model:")
                print_memory_usage()
                cpu_arima_model = getArimaModel(cpulist, cpu_trainRatio, 5, 5)
                print("After loading model:")
                print_memory_usage()
            except Exception:
                continue
            hour=0
            start_time=time()
            for startidx in range(cpu_startIdx, len(cpulist)-w+1):
                if hour==24:
                    break
                try:
                    next_cpu = forecastWithModel(cpu_arima_model, startidx, w)
                    # print(f'next cpu is {next_cpu}')
                except Exception as e:
                    print(f"ARIMA 预测时出现错误：{str(e)}")
                    continue
                
                real_cpu = cpulist[startidx:startidx+w]
                if len(next_cpu) != len(real_cpu):
                    continue
                cpu_mape = MAPE(next_cpu, real_cpu)
                if not np.isnan(cpu_mape):
                    np_cpu_mapelist.append(cpu_mape)
                hour+=1
            end_time=time()
            print(f'{cid}: 非周期性cpu，处理时间{end_time-start_time}, 平均处理时间{(end_time-start_time)/hour}')
        if cid in nonperiodic_mem:
            start_time=time()
            try:
                print("Before loading model:")
                print_memory_usage()
                mem_arima_model = getArimaModel(memlist, mem_trainRatio, 5, 5)
                print("After loading model:")
                print_memory_usage()
            except Exception:
                continue
            hour=0
            start_time=time()
            for startidx in range(mem_startIdx, len(memlist)-w+1):
                if hour==24:
                    break
                try:
                    next_mem = forecastWithModel(mem_arima_model, startidx, w)
                    # print(f'next mem is {next_mem}')
                except Exception as e:
                    print(f"ARIMA 预测时出现错误：{str(e)}")
                    continue
                
                real_mem = memlist[startidx:startidx+w]
                # print(f'{hour}:real_mem is {real_mem}, next mem is {next_mem}')
                if len(next_mem) != len(real_mem):
                    continue
                mem_mape = MAPE(next_mem, real_mem)
                # print(f'{hour}:mem_mape is {mem_mape}')
                if not np.isnan(mem_mape):
                    np_mem_mapelist.append(mem_mape)
                hour+=1
            end_time=time()
            print(f'{cid}: 非周期性mem，处理时间{end_time-start_time}, 平均处理时间{(end_time-start_time)/hour}')
        if cpu_mapelist:
            cpuMape=np.mean(np.array(cpu_mapelist))
            print(f'{cid}: 周期性cpu的mape为{cpuMape}')
            cpuMapeList.append(cpuMape)
        if mem_mapelist:
            memMape=np.mean(np.array(mem_mapelist))
            print(f'{cid}: 周期性mem的mape为{memMape}')
            memMapeList.append(memMape)
        if np_cpu_mapelist:
            cpuMape=np.mean(np.array(np_cpu_mapelist))
            print(f'{cid}: 非周期性cpu的mape为{cpuMape}')
            np_cpuMapeList.append(cpuMape)
        if np_mem_mapelist:
            memMape=np.mean(np.array(np_mem_mapelist))
            print(f'{cid}: 非周期性mem的mape为{memMape}')
            np_memMapeList.append(memMape) 
        with open('./preiodicAcc_arima.txt','w') as file:
            file.write(f'count={count}\n')
            file.write(f'在w为{w}时，周期性cpu的平均mape为{np.mean(np.array(cpuMapeList))}\n')
            file.write(f'在w为{w}时，周期性mem的平均mape为{np.mean(np.array(memMapeList))}\n')
            file.write(f'在w为{w}时，非周期性cpu的平均mape为{np.mean(np.array(np_cpuMapeList))}\n')
            file.write(f'在w为{w}时，非周期性mem的平均mape为{np.mean(np.array(np_memMapeList))}')
        count+=1


# if cid in periodic_cpu:
        #     # start=0
        #     for end in range(n,24*4-w):
        #         if end+w>=len(cpulist):
        #             break
        #         try:
        #             with warnings.catch_warnings():
        #                 warnings.simplefilter("ignore")
        #                 # next_cpu = arima_predict(cpulist, end, w)
        #                 next_cpu=ARIMA(cpulist, end, w, 5, 5)
        #         except Exception as e:
        #             print(f"ARIMA 预测时出现错误：{str(e)}")
        #             continue

        #         real_cpu=cpulist[end:end+w]
        #         if len(next_cpu)!=len(real_cpu):
        #             continue
        #         # 计算MAPE
        #         cpu_mape=MAPE(next_cpu,real_cpu)
        #         if not np.isnan(cpu_mape):
        #             cpu_mapelist.append(cpu_mape)
        #         # start+=1  
        # elif cid in periodic_mem:
        #     # start=0
        #     for end in range(n,24*4-w):
        #         if end+w>=len(memlist):
        #             break
        #         try:
        #             with warnings.catch_warnings():
        #                 warnings.simplefilter("ignore")
        #                 # next_mem = arima_predict(memlist, end, w)
        #                 next_mem = ARIMA(memlist, end, w, 5, 5)
        #         except Exception as e:
        #             print(f"ARIMA 预测时出现错误：{str(e)}")
        #             continue

        #         real_mem=memlist[end:end+w]
        #         if len(next_mem)!=len(real_mem):
        #             continue
        #         # 计算MAPE
        #         mem_mape=MAPE(next_mem,real_mem)
        #         if not np.isnan(mem_mape):
        #             mem_mapelist.append(mem_mape)
        #         # start+=1  