import numpy as np
import pandas as pd

# 计算每台机器上资源使用量
# 输入为t时刻各VM的cpu需求量以及VM放置情况
# 输出为t时刻各机器的资源使用总量
def ResourceUsage(cpuormem_t, x_t):
    """计算每台机器上资源使用量

    Args:
        cpuormem_t (numpy.ndarray): t时刻各VM的cpu或mem需求量
        x_t (numpy.ndarray): t时刻VM放置情况 N*M (vm N个, pm M个)

    Returns:
        float: t时刻各机器的资源使用总量
    """
    x_cput = x_t.copy().T
    CPU_t = np.matmul(x_cput,cpuormem_t)
    
    return CPU_t

# 适应v2022的版本
def ResourceUsage1(cpuormem_t, x_t):
    CPU_t=np.zeros(len(x_t))
    for i in range(len(x_t)):
        containerids=x_t.get(i+1,[])
        if containerids==['None'] or containerids==[]:
            CPU_t[i]=0
        else:
            idxs=[i-1 for i in containerids]
            CPU_t[i]=sum(cpuormem_t[idxs])
    return CPU_t



def isAllUnderLoad(x_t, cpu_t, mem_t, CPU_MAX, MEM_MAX):
    """判断所有机器是否过载

    Args:
        x_t0 (numpy.ndarray): t0时刻VM部署情况 N*M (vm N个, pm M个)
        cpu_t0 (numpy.ndarray): N个VM当前时刻的cpu需求量
        mem_t0 (numpy.ndarray): N个VM当前时刻的cpu需求量
        CPU_MAX (int): cpu资源使用上限
        MEM_MAX (int): mem资源使用上限

    Returns:
        bool: Ture or Not
    """
    CPU_t = ResourceUsage(cpu_t, x_t)
    MEM_t = ResourceUsage(mem_t, x_t)
    print("---- test sandpiper ----",CPU_t,MEM_t)
    is_cpu = (CPU_t < CPU_MAX).all()
    is_mem = (MEM_t < MEM_MAX).all()
    is_all = is_cpu and is_mem # 所以资源都不过载才返回True
    
    return is_all

def isAllUnderLoad1(x_t, cpu_t, mem_t, CPU_MAX, MEM_MAX):
    """判断所有机器是否过载

    Args:
        x_t0 (numpy.ndarray): t0时刻VM部署情况 N*M (vm N个, pm M个)
        cpu_t0 (numpy.ndarray): N个VM当前时刻的cpu需求量
        mem_t0 (numpy.ndarray): N个VM当前时刻的cpu需求量
        CPU_MAX (int): cpu资源使用上限
        MEM_MAX (int): mem资源使用上限

    Returns:
        bool: Ture or Not
    """
    CPU_t = ResourceUsage1(cpu_t, x_t)
    MEM_t = ResourceUsage1(mem_t, x_t)

    CPU_t_max=max(CPU_t)
    CPU_t_min=min(CPU_t)
    CPU_t=100*(CPU_t-CPU_t_min)/(CPU_t_max-CPU_t_min)
    # print(f'CPU_t max is {CPU_t_max}, CPU_t min is {CPU_t_min}')
    MEM_t_max=max(MEM_t)
    MEM_t_min=min(MEM_t)
    MEM_t=100*(MEM_t-MEM_t_min)/(MEM_t_max-MEM_t_min)
    # print(f'MEM_t max is {MEM_t_max}, MEM_t min is {MEM_t_min}')
    # print(f'after normalize, CPU_t max is {max(CPU_t)}, CPU_t min is {min(CPU_t)}, MEM_t max is {max(MEM_t)}, MEM_t min is {min(MEM_t)}')
    is_cpu = (CPU_t < CPU_MAX).all()
    is_mem = (MEM_t < MEM_MAX).all()
    is_all = is_cpu and is_mem # 所以资源都不过载才返回True
    
    return is_all

# 记录初始轨迹
def recordInitTrace(df : pd.DataFrame):
    trace={}
    for index,row in df.iterrows():
        containerid=row['container_id']
        machineid=row['machine_id']
        if containerid not in trace:
            trace[containerid]=[]

        trace[containerid].append(machineid)         
    return trace

# 记录轨迹【可以给df设置一些索引，加速】
def recordTrace(trace : dict, x_t : dict, df: pd.DataFrame):
    for new_machine_num,new_container_ids in x_t.items():
        # 寻找new_machine_num对应的machine_id
        # row=df.loc[new_machine_num].iloc[0]
        row=df[df['new_machine_num']==new_machine_num].iloc[0]
        machineid=row['machine_id']
        for ncid in new_container_ids:
            # 找到真正的containerid
            # row=df.loc[ncid]
            row=df[df['new_container_id']==ncid]
            containerid=row['container_id'].item() 

            if containerid not in trace:
                trace[containerid]=[machineid]
                continue

            if trace[containerid][-1]==machineid: # 本回合没有做迁移
                continue

            trace[containerid].append(machineid)
    
    return trace

def recordTrace_pro(trace : dict, mapping : dict):
    for machineid,container_ids in mapping.items():
        if not container_ids:
            continue
        if container_ids[0].startswith('None'):
            container_ids=[]
            continue
        for containerid in container_ids:
            if containerid not in trace:
                trace[containerid]=[machineid]
                continue

            if trace[containerid][-1]==machineid: # 本回合没有做迁移
                continue

            trace[containerid].append(machineid)
    return trace

def getInvalidMigrationNum(trace : dict):
    res=0
    for containerid, nodeids in trace.items():
        myset=set(nodeids)
        res+=len(nodeids)-len(myset)
    return res

# 得到经过两个及以上的node节点的无效迁移数
def getNumOfLongTravel(trace : dict):
    res=0
    for containerid, nodeids in trace.items():
        for i in range(len(nodeids)):
            for j in range(i+3, len(nodeids)):
                if nodeids[j]==nodeids[i]:
                    res+=1
                    break
    return res

# 记录迁移总数
def getTotalMigrationNum(trace : dict):
    res=0
    for containerid, nodeids in trace.items():
        res+=len(nodeids)
    return res


# 记录容器运行时间 【发现trace中竟然还有同一时刻中出现两次的重复数据，所以做了一点处理】
def getContainerUptime(file):
    uptime_dict={}
    read_chunks=pd.read_csv(file, iterator=True, chunksize=65535)
    for df_chunk in read_chunks:
        for minute in range(0, 24*60):
            df_filtered=df_chunk[df_chunk['time_stamp']==minute*60*1000]
            for containerid in df_filtered['container_id']:
                if containerid not in uptime_dict:
                    uptime_dict[containerid]=1
                else:
                    uptime_dict[containerid]+=1
    return uptime_dict

# placement转mapping
def getMapping(placement: dict, df: pd.DataFrame) -> dict:
    mapping = {}
    
    for new_nodeid, new_containeridList in placement.items():
        # 找到符合条件的行
        row = df.loc[df['new_machine_num'] == new_nodeid].iloc[0]
        machineid = row['machine_id']
        
        if machineid not in mapping:
            mapping[machineid] = []
        
        if new_containeridList == ['None']:
            continue
        
        for ncid in new_containeridList:
            # 找到符合条件的行
            row = df.loc[df['new_container_id'] == ncid]
            cid = row['container_id'].item()  # 假设只有一个匹配项，直接取值
            mapping[machineid].append(cid)

    return mapping

def getMapping_Pro(placement: dict, df: pd.DataFrame) -> dict:
    mapping = {}
    
    # 创建从new_nodeid到machineid的映射
    nodeid_to_machineid = df.set_index('new_machine_num')['machine_id'].to_dict()
    # 创建从new_containerid到containerid的映射
    newcid_to_cid = {}
    # 将 sorted_df_filtered 加载到字典中
    for idx, row in df.iterrows():
        newcid_to_cid[row['new_container_id']] = row['container_id']
    
    for new_nodeid, new_containeridList in placement.items():
        machineid = nodeid_to_machineid.get(new_nodeid)
        if not new_containeridList or new_containeridList==['None']:
            mapping[machineid]=[]
            continue
        
        if machineid is None:
            continue
        
        # 收集该new_nodeid对应的所有container_id
        container_ids = [newcid_to_cid[ncid] for ncid in new_containeridList]
        
        if machineid not in mapping:
            mapping[machineid] = []
        
        mapping[machineid].extend(container_ids)
    
    return mapping

# uptime_dict=getContainerUptime('./cnmap.csv')
# uptime_dict_order=sorted(uptime_dict.items(), key=lambda x:x[1], reverse=True)

# print(uptime_dict_order[0:100])
    