import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 设置 TensorFlow 日志级别为 ERROR 或更高级别
from lstm import getLstmModel, lstm_forecast_with_model, getLstmModel1
import ijson
import numpy as np
import psutil
from time import time

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
        # if count<=695:
        #     count+=1
        #     continue
        print(f'count is {count}')
        if count==1000:
            break
        cpu_list = [float(util[0]) for util in util_data if util[0] is not None]
        mem_list = [float(util[1]) for util in util_data if util[1] is not None]
        # 清理数据，去除无效值
        cpulist = [x for x in cpu_list if not np.isnan(x) and np.isfinite(x)]
        memlist = [x for x in mem_list if not np.isnan(x) and np.isfinite(x)]

        cpu_mapelist=list()
        mem_mapelist=list()
        np_cpu_mapelist=list()
        np_mem_mapelist=list()

        cpu_trainRatio=0.7
        cpu_startIdx=int(len(cpulist)*cpu_trainRatio)
        if cpu_startIdx==len(cpulist):
            cpu_startIdx=int(cpu_startIdx*0.7)
        
        mem_trainRatio=0.7
        mem_startIdx=int(len(memlist) * mem_trainRatio)
        if mem_startIdx==len(memlist):
            mem_startIdx=int(mem_startIdx*0.7)
        
        if cid in periodic_cpu:
            start_time=time()
            try:
                print("Before loading model:")
                print_memory_usage()
                cpu_model, cpu_scaler = getLstmModel1(cpulist, cpu_trainRatio, w)
                print("After loading model:")
                print_memory_usage()
            except Exception as e:
                continue
            end_time=time()
            print(f'{cid}: 周期性cpu，lstm训练时间{end_time-start_time}')
            hour=0
            start_time=time()
            for startidx in range(cpu_startIdx, len(cpulist) - 2 * w +1):
                if hour==24:
                    break
                try:
                    next_cpu = lstm_forecast_with_model(cpu_model, cpu_scaler, cpulist[startidx:startidx + w], w, w)
                except Exception as e:
                    print(f"LSTM 预测时出现错误：{str(e)}")
                    continue
                real_cpu = cpulist[startidx + w:startidx + 2 * w]
                if len(next_cpu) != len(real_cpu):
                    continue
                cpu_mape = MAPE(next_cpu, real_cpu)
                if not np.isnan(cpu_mape):
                    cpu_mapelist.append(cpu_mape)
                hour+=1
            end_time=time()
            print(f'{cid}: 周期性cpu，处理时间{end_time-start_time}, 平均处理时间{(end_time-start_time)/hour}')
        if cid in periodic_mem:
            start_time=time()
            try:
                print("Before loading model:")
                print_memory_usage()
                mem_model, mem_scaler = getLstmModel1(memlist, mem_trainRatio, w)
                print("After loading model:")
                print_memory_usage()
            except Exception as e:
                continue
            end_time=time()
            print(f'{cid}: 周期性mem，lstm训练时间{end_time-start_time}')
            hour=0
            start_time=time()
            for startidx in range(mem_startIdx, len(memlist) - 2 * w +1):
                if hour==24:
                    break
                try:
                    next_mem = lstm_forecast_with_model(mem_model, mem_scaler, memlist[startidx:startidx + w], w, w)
                except Exception as e:
                    print(f"LSTM 预测时出现错误：{str(e)}")
                    continue
                real_mem = memlist[startidx + w:startidx + 2 * w]
                if len(next_mem) != len(real_mem):
                    continue
                mem_mape = MAPE(next_mem, real_mem)
                if not np.isnan(mem_mape):
                    mem_mapelist.append(mem_mape)
                hour+=1
            end_time=time()
            print(f'{cid}: 周期性mem，处理时间{end_time-start_time}, 平均处理时间{(end_time-start_time)/hour}')
        if cid in nonperiodic_cpu:
            start_time=time()
            try:
                print("Before loading model:")
                print_memory_usage()
                cpu_model, cpu_scaler = getLstmModel1(cpulist, cpu_trainRatio, w)
                print("After loading model:")
                print_memory_usage()
            except Exception as e:
                continue
            end_time=time()
            print(f'{cid}: 非周期性cpu，lstm训练时间{end_time-start_time}')
            hour=0
            start_time=time()
            for startidx in range(cpu_startIdx, len(cpulist) - 2 * w +1):
                if hour==24:
                    break
                try:
                    next_cpu = lstm_forecast_with_model(cpu_model, cpu_scaler, cpulist[startidx:startidx + w], w, w)
                except Exception as e:
                    print(f"LSTM 预测时出现错误：{str(e)}")
                    continue
                real_cpu = cpulist[startidx + w:startidx + 2 * w]
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
                mem_model, mem_scaler = getLstmModel1(memlist, mem_trainRatio, w)
                print("After loading model:")
                print_memory_usage()
            except Exception as e:
                continue
            end_time=time()
            print(f'{cid}: 非周期性mem，lstm训练时间{end_time-start_time}')
            hour=0
            start_time=time()
            for startidx in range(mem_startIdx, len(memlist) - 2 * w +1):
                if hour==24:
                    break
                try:
                    next_mem = lstm_forecast_with_model(mem_model, mem_scaler, memlist[startidx:startidx + w], w, w)
                except Exception as e:
                    print(f"LSTM 预测时出现错误：{str(e)}")
                    continue
                real_mem = memlist[startidx + w:startidx + 2 * w]
                if len(next_mem) != len(real_mem):
                    continue
                mem_mape = MAPE(next_mem, real_mem)
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
        count+=1

        with open('./preiodicAcc_lstm.txt','w') as file:
            file.write(f'count is {count}')
            file.write(f'在w为{w}时，周期性cpu的平均mape为{np.mean(np.array(cpuMapeList))}\n')
            file.write(f'在w为{w}时，周期性mem的平均mape为{np.mean(np.array(memMapeList))}\n')
            file.write(f'在w为{w}时，非周期性cpu的平均mape为{np.mean(np.array(np_cpuMapeList))}\n')
            file.write(f'在w为{w}时，非周期性mem的平均mape为{np.mean(np.array(np_memMapeList))}')