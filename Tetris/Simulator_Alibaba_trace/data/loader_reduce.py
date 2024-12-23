from email import header
import pandas as pd
import os
import csv
import numpy as np
import pandas as pd
from instance import InstanceConfig
from machine import MachineConfig
import random
# import dask.dataframe as dd

"""
[100,1700] [500,8500] [1000,17000] [2000,34000] [3000,51000] [3989,67437]
"""
def read_iterator(filepath):
    cpulist = {}
    memlist = {}
    files = os.listdir(filepath)
    # n = int(800)
    
    for idx, file in enumerate(files):
        filename = os.path.join(filepath, file) # 
        ids = int(filename[filename.rfind('_')+1:filename.rfind('.')])
        
        df = pd.read_csv(filename,header=None)
        # cpus = cpus.get_chunk(n)
        # df = pd.read_csv(f,header=None)
        # df.rename(columns={0:'cpu' ,1:'mem'},inplace=True)
        cpu = df[0].values.tolist()
        mem = df[1].values.tolist() # 这里报错，没有df[1]
        cpulist[ids] = cpu
        memlist[ids] = mem
        # print(idx,',',filename,', cpu list: ',len(cpulist))
    return cpulist, memlist


# @profile
# 根据给定的虚拟机CPU请求文件和节点数、容器数的列表，加载实例配置信息，并将结果以列表的形式返回
# test_array [[节点数, 容器数], ...]
def InstanceConfigLoader(vm_cpu_request_file,test_array):
    res = []
    instance_configs = {} # 存储实例配置信息
    inc_mac_id_file = 'container_machine_id.csv' # 容器与机器对应关系的文件路径
    vm_mac = {} # 这里没用到这个变量
    machine_configs = {} # 存储机器配置信息 【machine_id : MachineConfig】
    # 读取所有vm的资源
    vm_cpu_requests, vm_mem_requests = read_iterator(vm_cpu_request_file)
    mac = {} # {machine_id : [inc_id, ...]}, 表示机器中有哪些容器 
    # 读取第一时刻vm安置的关系
    df = pd.read_csv(inc_mac_id_file, header=None)
    # 存储 container【新id ： 旧id】
    inc_ids = {}
    # 存储 machine 【新id：旧id】
    mac_ids = {}
    '''
    由于最初的机器对应container的id表有些是是整数不连续的,为了方便后面计算,换成连续
    存储为mac_new
    vm_cpu_requests , vm_mem_requests 里container对应的是不连续的id
    '''
    for idx, data in df.iterrows():
        '''inc_id 是从0开始的连续整数
           data[0] 是container id 从1开始的非连续整数序列
           data[1] 是machine id 从1开始的非连续整数序列

        '''
        inc_id = idx
        inc_ids[inc_id] = data[0]
        mac_id = data[1]
        
        if mac_id in mac:
            mac[mac_id].append(inc_id)
        else:
            mac[mac_id] = [inc_id]

    mac = {k: v for k, v in sorted(mac.items(), key=lambda x: x[0])} # 按照键machine id进行排序
    # print(mac)
    # mac_new machine id 连续
    mac_new = {}
    idx = 0
    # 将mac重新整合到mac_new中
    for k, v in mac.items():
        # print(f'idx={idx} newmac_id={k}')
        mac_new[idx] = v
        mac_ids[idx] = k # 存储machine的【新id：旧id】
        idx = idx+1
    # mac = mac_new
    machine_half = {} # 存储一半的机器配置
    
    import random
    
    fileName = "3989.csv"
    dataframe= pd.read_csv(fileName)
    
    # 根据test_array中的每个元组，生成机器和实例的配置信息
    for tup in test_array:
        nodeNum = tup[0]
        containerNum = tup[1]
        mac_nodes = dataframe["macid"].values.tolist()[:nodeNum] # 读取前nodeNum个macid
        mac = { k:mac_new[k]for k in  mac_nodes } # 【macid：[inc_id, ...]】
        print(f'len of mac is len(mac)')
        print(mac)
        # summac = sum([ len(mac[mac_nodes[i]]) for i in range(nodeNum)])
        summac = sum([ len(v) for v in mac.values()]) # 所有机器上的容器数量
        print(nodeNum,containerNum,summac)
        
        for machine_id, data in mac.items():
            # 生成machine；machine_configs [mac_id:machine实体类]
            machine = MachineConfig(machine_id, 30, 100)
            machine_configs[machine_id] = machine
            
            for instanceid in data:
                if machine_id not in machine_half:
                    machine_half[machine_id] = [instanceid]
                else:
                    machine_half[machine_id].append(instanceid)
                # 根据旧容器id，获取对应的CPU请求曲线、内存请求曲线
                cpu_curve = vm_cpu_requests[inc_ids[instanceid]]
                # test_csv(str(inc_ids[instanceid]),cpu_curve)
                memory_curve = vm_mem_requests[inc_ids[instanceid]]
                disk_curve = np.zeros_like(cpu_curve)
                instance_config = InstanceConfig(
                    machine_id, instanceid, cpu_curve[0], memory_curve[0], disk_curve, cpu_curve, memory_curve)
                instance_configs[instanceid] = instance_config
        print(f'len of instance_configs is {len(instance_configs)}')
        i = 0
        j =0
        new_machins  = {}
        new_instances = {}
        
        for macid,inslist in mac.items():
            mac = machine_configs[macid]
            mac.id = j
            new_machins[j] = mac
            j+=1
            
            for incid in inslist:
                inc = instance_configs[incid]
                inc.machine_id = mac.id
                inc.id = i
                new_instances[i] = inc
                i+=1
        print("half_data instance number",len(new_instances),"half machine number ",len(new_machins))
        res.append([new_instances, new_machins, mac_ids,inc_ids])

    print(f'len of res is {len(res)}')   
    return res


def test_csv(old_id,cpu_curve):
    path = '/hdd/jbinin/AlibabaData/target/'
    filename = 'instanceid_'+old_id+'.csv'
    instance_path = os.path.join(path,filename)
    df = pd.read_csv(instance_path,header=None)[0][:10].values
    cpulist = np.array(cpu_curve[0:10])
    
    print(old_id,df,cpulist)
    assert (df==cpulist).all()