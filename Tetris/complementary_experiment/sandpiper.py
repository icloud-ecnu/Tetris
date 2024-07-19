from time import time                                                                                                 
from typing import overload           
import numpy as np
from utl import ResourceUsage1,isAllUnderLoad1

# 适应v2022版本的Sandpiper_algo, x_t0不再是N×M，因为存储不了
def Sandpiper_algo1(x_t0, cpu_t0, mem_t0, CPU_MAX, MEM_MAX):
    
    f = isAllUnderLoad1(x_t0, cpu_t0, mem_t0, CPU_MAX, MEM_MAX)
    if f :
        print("no schedule")
        return x_t0
    
    # 计算当前各PM和VM的Vol值及VSR值
    CPU_t0 = ResourceUsage1(cpu_t0, x_t0)
    MEM_t0 = ResourceUsage1(mem_t0, x_t0)
    
    CPU_t0_max=max(CPU_t0) + 1
    CPU_t0_min=min(CPU_t0)
    CPU_t0=100*(CPU_t0-CPU_t0_min)/(CPU_t0_max-CPU_t0_min)

    MEM_t0_max=max(MEM_t0) + 1
    MEM_t0_min=min(MEM_t0)
    MEM_t0=100*(MEM_t0-MEM_t0_min)/(MEM_t0_max-MEM_t0_min)
    # print(f'algo1: after normalize, CPU_t max is {max(CPU_t0)}, CPU_t min is {min(CPU_t0)}, MEM_t max is {max(MEM_t0)}, MEM_t min is {min(MEM_t0)}')
    # print(f'len of CPU_t0 is {len(CPU_t0)}, len of MEM_t0 is {len(MEM_t0)}')
    Vol_pm = 10000 / ((100 - CPU_t0) * (100 - MEM_t0)) # 注意每PM/VM的资源需求要<100%
    # print(f'Vol_pm is {Vol_pm}')
    Vol_vm = 10000 / ((100 - cpu_t0) * (100 - mem_t0)) # 1*N矩阵 ？？？？
    VSR = Vol_vm/ mem_t0 # 1*N矩阵
    # 机器按Vol值按序排序，存储机器号
    # 那些空的node对应的Vol_pm是1，排序越小
    pm_asc = Vol_pm.argsort()
    # print(f'pm_asc is {pm_asc}, len is {len(pm_asc)}, max is {pm_asc.max()}, min is {pm_asc.min()}')
    pm_desc = pm_asc[::-1]
    #print('pm_desc',pm_desc)
    placement = x_t0.copy() # 初始化
    # 按序对每台机器做迁出
    pm_asc.astype(int)
    pm_desc.astype(int)
    migs_inc_outToin = {}
    pmout = {}
    pmin={}
    #print(f'\t to schedule pm num is {len(pm_desc)}')
    idx=0
    motivation = {}
    test_cpu = set()
    test_mem = set()
    metric_pmout = set()
    metric_pmin = set()
    # print(f'pm_desc is {pm_desc}')                                                                                                                                                                                       
    for pm_outs in pm_desc: # 越满的node越先处理
        pm_out=int(pm_outs)
        #print('\t\t sand piper',idx)                                                                                                                                                                         
        idx+=1
        if CPU_t0[pm_out] <= CPU_MAX and MEM_t0[pm_out] <= MEM_MAX: # 按序迁出该机器上的VM直到机器不过载
            continue
        # 将每台机器上的VM降序排序，存储VM号
        # vm_in_pm = np.where(x_t0[:, pm_out] == 1)[0] # 该机器上VM的VM号
        vm_in_pm = x_t0[pm_out+1] # 该机器上VM的VM号
        if vm_in_pm==['None'] or vm_in_pm==[]:
            continue
        r_idx=[x-1 for x in vm_in_pm]
        VSR_in_pm = VSR[r_idx] # 这些VM的VSR
        # print(vm_in_pm,VSR_in_pm)
        vm_VSR = np.array([vm_in_pm, VSR_in_pm]) # 二维数组，第一行为VM号，第二行为VM对应VSR值
        vm_asc = vm_VSR.T[np.lexsort(vm_VSR)].T # 按照VSR升序排序
        #vm_asc = vm_VSR[:,vm_VSR[1].argsort()]
        vm_desc = vm_asc[0, ::-1] # 获取降序排序后的VM号
        vm_desc.astype(int)
        for vms in vm_desc: # 从VSR最大的VM开始被迁移
            vm = int(vms)
            if CPU_t0[pm_out] <= CPU_MAX and MEM_t0[pm_out] <= MEM_MAX: # 按序迁出该机器上的VM直到机器不过载
                break
            for pm_inx in pm_asc: # 从Vol最小的开始迁入
                pm_in = int(pm_inx)
                if CPU_t0[pm_in] + cpu_t0[vm-1] <= CPU_MAX and MEM_t0[pm_in] + mem_t0[vm-1] <= MEM_MAX: # 有机器放得下
                    metric_pmout.add(pm_out)
                    metric_pmin.add(pm_in)
                    s = time()
                    # placement[vm][pm_out] = 0 # 迁出机器
                    if placement[pm_out+1]!=['None']:
                        placement[pm_out+1].remove(vm)
                        if placement[pm_out+1]==[]:
                            placement[pm_out+1]=['None']
                        # print(f'{pm_out+1} migrate out {vm}')
                            
                    CPU_t0[pm_out] = CPU_t0[pm_out] - cpu_t0[vm-1]
                    MEM_t0[pm_out] = MEM_t0[pm_out] - mem_t0[vm-1]
                    # placement[vm][pm_in] = 1 # 迁入机器
                    if placement[pm_in+1]==['None']:
                        placement[pm_in+1]=[vm]
                    else:
                        placement[pm_in+1].append(vm)
                    # print(f'{pm_in+1} migrate in {vm}')

                    CPU_t0[pm_in] = CPU_t0[pm_in] + cpu_t0[vm-1]
                    MEM_t0[pm_in] = MEM_t0[pm_in] + mem_t0[vm-1]
                    if pm_out in pmout:
                        pmout[pm_out].append(vm)
                    else:
                        pmout[pm_out]=[vm]
                    if pm_in in pmin:
                        pmin[pm_in].append(vm)
                    else:
                        pmin[pm_in]=[vm]
                    migs_inc_outToin[vm]=(pmout,pmin)
                    break
        # 循环结束条件是没有机器过载
        if (CPU_t0 <= CPU_MAX).all() and (MEM_t0 <= MEM_MAX).all():
            #f = isAllUnderLoad(placement, cpu_t0, mem_t0, CPU_MAX, MEM_MAX)
            #print(f'\t\tall pm is underload ?={f}')
            break

    return placement

