import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import ijson

from statsmodels import api as sm
from scipy.stats import pearsonr
import scipy.signal as signal

def isPeriodic(tempNorm):
    periodFlag = 0
    try:
        acf = sm.tsa.acf(tempNorm, nlags=len(tempNorm)) # 计算自相关系数

        peak_ind = signal.argrelextrema(acf, np.greater)[0] # 寻找局部峰值
        fwbest = acf[signal.argrelextrema(acf, np.greater)]

        index = -1
        ran = 0
        fwbestlen = len(fwbest)
        if fwbestlen == 0:
            periodFlag = 0
            return periodFlag
        
        for i in range(ran, fwbestlen):
            if fwbest[i] > 0:
                j = i
                while fwbest[j] > 0:
                    j += 1
                    if j > fwbestlen - 1:
                        periodFlag = 1
                        return periodFlag
                index = (i + j - 1) // 2
                break

        fd = peak_ind[index] # 频率
        numlist = []
        Q = len(tempNorm) // fd # 周期
        if Q == 1:
            periodFlag = 0
            return periodFlag
        else:
            for i in range(Q): # 分段
                numlist.append(tempNorm[i * fd: (i + 1) * fd])

            listlen = len(numlist) # 段数
            flag = 0
            for i in range(1, listlen):
                std_prev = np.std(numlist[i-1])
                std_curr = np.std(numlist[i])
                if std_prev > 1e-6 and std_curr > 1e-6:  # 检查标准差是否足够大
                    a = pearsonr(numlist[i-1], numlist[i])[0]  # 相邻两段的皮尔森系数
                    if a < 0.85:
                        flag += 1  # 小于阈值的数量

            if flag <= listlen // 3: # 小于阈值的低于总段数的1/3
                periodFlag = 1
                return periodFlag
            else:
                periodFlag = 0
                return periodFlag

    except Exception as e:
        print(f"处理数据时出现错误：{str(e)}")
        return periodFlag
    
# 曲线平滑处理
def Smooth(ts):
    print(f'ts is {ts}')
    dif = ts.diff().dropna() # 差分序列，1-69119行
    print(f'dif is {dif}')
    td = dif.describe() # 描述性统计得到：min，25%，50%，75%，max值
    print(f'td is {td}')

    high = td['75%'] + 1.5 * (td['75%'] - td['25%']) # 定义高点阈值，1.5倍四分位距之外
    low = td['25%'] - 1.5 * (td['75%'] - td['25%']) # 定义低点阈值，同上
    print(f'high is {high}, low is {low}')
    # 变化幅度超过阈值的点的索引
    forbid_index = dif[(dif > high) | (dif < low)].index 
    print(f'forbid_index is {forbid_index}')
    i = 0
    while i < len(forbid_index) - 1:
        n = 1 # 发现连续多少个点变化幅度过大，大部分只有单个点
        start = forbid_index[i] # 异常点的起始索引
        while (i+n)<len(forbid_index) and forbid_index[i+n] == start + n:
            n += 1
        i += n - 1
    
        end = forbid_index[i] # 异常点的结束索引
        print(f'start is {start}, end is {end}')
        # 用前后值的中间值均匀填充
        value = np.linspace(ts[start-1], ts[end+1], n) # 创建等差数列
        print(f'value is {value}')
        ts[start:end+1] = value
        i += 1
        
    return ts

# nonperiodic_cpu = pd.DataFrame(columns=['id'])
# periodic_cpu = pd.DataFrame(columns=['id'])
# nonperiodic_mem = pd.DataFrame(columns=['id'])
# periodic_mem = pd.DataFrame(columns=['id'])

# containerUtilDict={}
# with open('containerUtilDict_1.txt', 'r') as file:
#     jsonstr=file.read()
#     containerUtilDict = json.loads(jsonstr)

# print('加载containerUtilDict成功')
# for cid, utilList in containerUtilDict.items():
#     print(f'utillist is {utilList}')
#     # 整合后面的utilList
#     print('get another dict')
#     for i in range(2,5):
#         with open(f'containerUtilDict_{i}.txt', 'r') as file:
#             jsonstr=file.read()
#             containerUtilDict_i=json.loads(jsonstr)
#             if not containerUtilDict_i[cid]:
#                 continue
#             else:
#                 utilList_i=containerUtilDict_i[cid]
#                 utilList=utilList+utilList_i
#     print(f'utillist is {utilList}')
#     cpu_list=[util[0] for util in utilList]
#     mem_list=[util[0] for util in utilList]

#     # 判断周期性
#     flag_cpu = isPeriodic(cpu_list)
#     # print(f'flag_cpu is {flag_cpu}')
#     if flag_cpu == 0:
#         nonperiodic_cpu.loc[len(nonperiodic_cpu)] = [cid]
#     elif flag_cpu == 1:
#         periodic_cpu.loc[len(periodic_cpu)] = [cid]
        
#     flag_mem = isPeriodic(mem_list)
#     # print(f'flag_mem is {flag_mem}')
#     if flag_mem == 0:
#         nonperiodic_mem.loc[len(nonperiodic_mem)] = [cid]
#     elif flag_mem == 1:
#         periodic_mem.loc[len(periodic_mem)] = [cid]


# # 打印结果或做进一步处理
# print("非周期性 CPU 容器:", nonperiodic_cpu)
# print("周期性 CPU 容器:", periodic_cpu)
# print("非周期性 内存 容器:", nonperiodic_mem)
# print("周期性 内存 容器:", periodic_mem)


nonperiodic_cpu = pd.DataFrame(columns=['id'])
periodic_cpu = pd.DataFrame(columns=['id'])
nonperiodic_mem = pd.DataFrame(columns=['id'])
periodic_mem = pd.DataFrame(columns=['id'])

count=1
with open('/mnt/d/school/tetris/containerUtilDict.txt', 'r') as file:
    parser=ijson.kvitems(file, '')
    print('get parser')
    for cid, util_data in parser:
        # if count==100000:
        #     break
        # print(f'Container ID: {cid}')
        cpu_list = [float(util[0]) for util in util_data if util[0] is not None]
        mem_list = [float(util[1]) for util in util_data if util[1] is not None]
        # print(f'len of cpu list is {len(cpu_list)}, cpu list is {cpu_list}')
        # print(f'len of mem list is {len(mem_list)}, mem list is {mem_list}')
        # 清理数据，去除无效值
        cpu_list = [x for x in cpu_list if not np.isnan(x) and np.isfinite(x)]
        mem_list = [x for x in mem_list if not np.isnan(x) and np.isfinite(x)]
        # print(f'len of cpu list is {len(cpu_list)}, cpu list is {cpu_list}')
        # print(f'len of mem list is {len(mem_list)}, mem list is {mem_list}')
        if not cpu_list or not mem_list:
            print(f'Container ID: {cid} has no valid CPU or memory data.')
            continue

        # 判断周期性
        flag_cpu = isPeriodic(cpu_list)
        # print(f'flag_cpu is {flag_cpu}')
        if flag_cpu == 0:
            nonperiodic_cpu.loc[len(nonperiodic_cpu)] = [cid]
        elif flag_cpu == 1:
            periodic_cpu.loc[len(periodic_cpu)] = [cid]
            
        flag_mem = isPeriodic(mem_list)
        # print(f'flag_mem is {flag_mem}')
        if flag_mem == 0:
            nonperiodic_mem.loc[len(nonperiodic_mem)] = [cid]
        elif flag_mem == 1:
            periodic_mem.loc[len(periodic_mem)] = [cid]
        count+=1
# 打印结果或做进一步处理
# print("非周期性 CPU 容器: \n", nonperiodic_cpu)
# print("周期性 CPU 容器: \n", periodic_cpu)
# print("非周期性 内存 容器: \n", nonperiodic_mem)
# print("周期性 内存 容器: \n", periodic_mem)
nonperiodic_cpu.to_csv('nonperiodic_cpu.csv',index=False)
periodic_cpu.to_csv('periodic_cpu.csv',index=False)
nonperiodic_mem.to_csv('nonperiodic_mem.csv',index=False)
periodic_mem.to_csv('periodic_mem.csv',index=False)
