import csv
import pandas as pd
import numpy as np

from sandpiper import Sandpiper_algo1


# 给文件排序
def sortcnmap(filepath):
    with open(filepath,'r') as file:
        csv_reader=csv.reader(file)
        header = next(csv_reader)  # 跳过第一行标题行
        sorted_rows=sorted(csv_reader, key=lambda x: float(x[0]))
        
    with open('sorted_cnmap.csv','w',newline='') as file:
        csv_writer=csv.writer(file)
        csv_writer.writerow(header)  # 写入标题行
        csv_writer.writerows(sorted_rows)

# container_id 排序规则
def custom_sort_rule(x):
    xstrArray=x.str.split('_')
    x1 = xstrArray.str[1].astype(float)
    x2 = xstrArray.str[3].astype(float)
    return x1+x2/1000.0

# 改进版，由于容器归一化的范围是以node为范围的，不同node上的容器数据不具备可比较性，所以考虑了node节点的cpu和mem利用率
# return: 
# cpu_t0, mem_t0: 以容器id为升序的对应的cpu和mem数组
# x_t0: {new_machine_num: [new_container_id]}, new_machine_num升序排列
def getMappingAndUtil_pro(filepath, node_filepath, time_stamp=0):
    df_chunks = pd.read_csv(filepath, iterator=True, chunksize=100000)
    df_list = [df_chunk[df_chunk['timestamp'] == time_stamp] for df_chunk in df_chunks]
    df_filtered = pd.concat(df_list)

    df_node = pd.read_csv(node_filepath)
    df_node_filtered = df_node[df_node['timestamp'] == time_stamp]
    if df_node_filtered.empty or df_filtered.empty:
        return [[],[],[],[],0]
    # 构造节点的cpu和mem的util的哈希字典
    node_dict = dict(zip(df_node_filtered['machine_id'], zip(df_node_filtered['cpu_util'], df_node_filtered['mem_util'])))

    sorted_df_filtered = df_filtered.sort_values('container_id', key=custom_sort_rule) # 使用自定义规则对container_id排序
    
    # print(f'after sort, sorted_df_filtered is:\n {sorted_df_filtered}')
    
    # 获取在 container_id 升序排序下的各个容器对应的 node 的 cpu 和 mem 使用量情况
    node_ids = sorted_df_filtered['machine_id'].values  # 获取所有节点的 machine_id

    # 只存储该时刻存在的node的cpu和mem利用率
    cpu_t0_node = []
    mem_t0_node = []
    bad_node_ids=[] # 记录那些此时刻在container数据中存在，但在node数据中不存在的nodeid
    for node_id in node_ids:
        # print(f'node_id is {node_id}')
        if node_id not in node_dict:
            bad_node_ids.append(node_id)
            continue

        cpu_util = node_dict[node_id][0]  # 获取节点的 CPU 利用率
        mem_util = node_dict[node_id][1]  # 获取节点的内存利用率
        cpu_t0_node.append(cpu_util)
        mem_t0_node.append(mem_util)

    cpu_t0_node = np.array(cpu_t0_node)
    mem_t0_node = np.array(mem_t0_node)
    # print(f'shape of cpu_t0_node is: {np.shape(cpu_t0_node)}')
    # print(f'shape of mem_t0_node is: {np.shape(mem_t0_node)}')
    # print(f'cpu_t0_node is: {cpu_t0_node}')
    # print(f'mem_t0_node is: {mem_t0_node}')
    # print(f'len of bad nodeids is: {len(bad_node_ids)}')

    # 去除那些坏的machine_id所在的行
    sorted_df_filtered = sorted_df_filtered[~sorted_df_filtered['machine_id'].isin(bad_node_ids)]
    # print(f'after delete bad nodeids, sorted_df_filtered is:\n {sorted_df_filtered}')

    # 给所有的containerid排序
    sorted_df_filtered['new_container_id'] = range(1, len(sorted_df_filtered) + 1)
    # print(f'after add new_container_id, sorted_df_filtered is:\n {sorted_df_filtered}')

    # 得到各个 container 的利用率【只获取那些对应node节点存在的container的利用率】
    cpu_t0 = sorted_df_filtered['cpu_util'].values
    mem_t0 = sorted_df_filtered['mem_util'].values
    # print(f'shape of cpu_t0 is: {np.shape(cpu_t0)}')
    # print(f'shape of mem_t0 is: {np.shape(mem_t0)}')
    # print(f'cpu_t0 is: {cpu_t0}')
    # print(f'mem_t0 is: {mem_t0}')

    # 提取出nodeid，并升序排序
    sorted_df_filtered['machine_num'] = sorted_df_filtered['machine_id'].str.extract(r'NODE_(\d+)', expand=False).astype(int)
    sorted_df_filtered = sorted_df_filtered.sort_values(by='machine_num', ascending=True)
    sorted_df_filtered['new_machine_num']=sorted_df_filtered['machine_num'].rank(ascending=True, method='dense').astype(int)
    # print(f'after add new_machine_num, sorted_df_filtered is:\n {sorted_df_filtered}')
    
    # 构建容器节点映射表 mapping_dict
    mapping_dict = sorted_df_filtered.groupby('new_machine_num')['container_id'].apply(list).to_dict()

    # 计算 container 的 cpu 和 mem 利用率
    assert len(cpu_t0_node) == len(cpu_t0) and len(mem_t0_node) == len(mem_t0)
    cpu_t0 = cpu_t0 * cpu_t0_node * 100 + 1
    mem_t0 = mem_t0 * mem_t0_node * 100 + 1

    # print(f'after compuate, real cpu_t0 is: {cpu_t0}')
    # print(f'after compuate, real mem_t0 is: {mem_t0}')


    # 再对cpu_t0，mem_t0做归一化，得带百分比
    cpu_t0_max=max(cpu_t0)+1
    cpu_t0_min=min(cpu_t0)-1
    # print(f'max cpu:{cpu_t0_max}, min cpu:{cpu_t0_min}')
    # cpu_t0=pd.Series(cpu_t0)
    # cpu_t0.apply(lambda x:MaxMinMethod(cpu_t0_max,cpu_t0_min,x)*100)
    cpu_t0=MaxMinMethod(cpu_t0_max,cpu_t0_min,cpu_t0)*100

    mem_t0_max=max(mem_t0)+1
    mem_t0_min=min(mem_t0)-1
    # print(f'max mem:{mem_t0_max}, min mem:{mem_t0_min}')
    # mem_t0=pd.Series(mem_t0)
    # mem_t0.apply(lambda x:MaxMinMethod(mem_t0_max,mem_t0_min,x)*100)
    mem_t0=MaxMinMethod(mem_t0_max,mem_t0_min,mem_t0)*100

    # print(f'after normalize, type of cpu_t0 is: {type(cpu_t0)}, cpu_t0 is: \n{cpu_t0}')
    # print(f'after normalize, type of mem_t0 is: {type(mem_t0)}, mem_t0 is: \n{mem_t0}')
    # print(f'after normalize, max cpu is {max(cpu_t0)}, min cpu is {min(cpu_t0)}, max mem is {max(mem_t0)}, min mem is {min(mem_t0)}')
    # 构建 x_t0 字典
    x_t0 = {new_machine_num: sorted_df_filtered[sorted_df_filtered['new_machine_num'] == new_machine_num]['new_container_id'].values.tolist()
            for new_machine_num in mapping_dict}

    # sorted_df_filtered.set_index(['new_machine_num', 'new_container_id'])
    return sorted_df_filtered, cpu_t0, mem_t0, x_t0, 1

# 获取placement对应的下一时刻的cpu和mem值
# mapping: {nodeid:[cid, cid ...], ...}
def getMappingUtilForPlacement(filepath, node_filepath, mapping:dict, trace:dict, time_stamp=0):
    df_chunks = pd.read_csv(filepath, iterator=True, chunksize=100000)
    df_list = [df_chunk[df_chunk['timestamp'] == time_stamp] for df_chunk in df_chunks]
    df_filtered = pd.concat(df_list)

    df_node = pd.read_csv(node_filepath)
    df_node_filtered = df_node[df_node['timestamp'] == time_stamp]

    # 构造节点的cpu和mem的util的哈希字典
    node_dict = dict(zip(df_node_filtered['machine_id'], zip(df_node_filtered['cpu_util'], df_node_filtered['mem_util'])))

    # 去重
    df_filtered = df_filtered.drop_duplicates(subset=['container_id'],keep='first')

    sorted_df_filtered = df_filtered.sort_values('container_id', key=custom_sort_rule) # 使用自定义规则对container_id排序
    
    # print(f'after sort, sorted_df_filtered is:\n {sorted_df_filtered}')
    
    # 获取在 container_id [升序排序]下的各个容器对应的 node 的 cpu 和 mem 使用量情况
    node_ids = sorted_df_filtered['machine_id'].values  # 获取所有节点的 machine_id

    # 只存储该时刻存在的node的cpu和mem利用率
    valid_node_ids = np.array([node_id for node_id in node_ids if node_id in node_dict])
    cpu_t0_node = np.array([node_dict[node_id][0] for node_id in valid_node_ids])
    mem_t0_node = np.array([node_dict[node_id][1] for node_id in valid_node_ids])
    bad_node_ids = list(set(node_ids) - set(valid_node_ids)) # 记录那些此时刻在container数据中存在，但在node数据中不存在的nodeid

    # 去除那些坏的machine_id所在的行
    sorted_df_filtered = sorted_df_filtered[~sorted_df_filtered['machine_id'].isin(bad_node_ids)]
    
    # print(f'after delete bad nodeids, sorted_df_filtered is:\n {sorted_df_filtered}')
    # 给所有的containerid编号
    sorted_df_filtered['new_container_id'] = range(1, len(sorted_df_filtered) + 1)
    # print(f'after add new_container_id, sorted_df_filtered is:\n {sorted_df_filtered}')
    
    # =============此刻，t时刻的trace中记录的容器按id升序排序，并且与cpu_t0_node、mem_t0_node一一对应==============
    # 但是，上一时刻的placement并不一定与此时刻的映射一致，可能多出或缺少一些容器，需进一步调整
    # 1、首先遍历mapping中的容器，若发现此时不存在了，那么视作此时刻关闭，不参与，移出mapping
    cids=set(sorted_df_filtered['container_id'].values.tolist())
    for nodeid, cidlist in mapping.items():
        cidlist[:] = [cid for cid in cidlist if cid in cids] # 这里会把key-['None']的项变为key-[]
    
    # 2、然后遍历容器，若发现新出现的，则视作新启动的容器，把其和其对应的node加入mapping
    containeridList=set(j for i in mapping.values() for j in i) # 使用set，加快查找速度（基于hash）
    # print(f'len containeridList is {len(containeridList)}')
    for cid in cids:
        if cid not in containeridList:
            # print(cid)
            nodeid=sorted_df_filtered[sorted_df_filtered['container_id']==cid]['machine_id'].item()
            if nodeid not in mapping:
                mapping[nodeid]=[]
            mapping[nodeid].append(cid)
            # 更新trace
            trace[cid]=[nodeid]
    # ============至此，mapping中的容器已经与此时的trace记录一致=====================mapping可能会有一些value为[]的项。表明一些节点上没有任何容器
    assert len([v for vlist in mapping.values() for v in vlist])==len(sorted_df_filtered)

    # print('---- finish mapping update ---')
    # sorted_df_filtered = sorted_df_filtered.reset_index(drop=True)
    # sorted_df_filtered = sorted_df_filtered.set_index(['container_id', 'machine_id', 'new_container_id'])
    # 建立所有元素的新的映射dataframe
    new_df=pd.DataFrame()
    containeridList=[cid for nid,cids in mapping.items() for cid in cids or ['None+'+nid]] # 会比trace中存在的容器数量多，因为有一些'None+xxx'的id存在
    index_mapping = {c: k for k, v in mapping.items() if v for c in v} # 对于空value会直接跳过

    new_df['container_id']=containeridList # ---container_id为None+xxx的，表明它对应的节点xxx上没有容器---
    # print(f'new_df is \n{new_df}')

    # 创建一个映射字典，将 container_id 映射到 new_container_id
    cid_to_new_cid = {}
    # 将 sorted_df_filtered 加载到字典中
    for idx, row in sorted_df_filtered.iterrows():
        cid_to_new_cid[row['container_id']] = row['new_container_id']
    # print('finish loading')
    # 一次性填充映射字典
    for cid in containeridList:
        if cid.startswith('None'):
            cid_to_new_cid[cid] = 'None'
        else:
            cid_to_new_cid[cid] = cid_to_new_cid.get(cid, 'None')
    # print('cid_to_new_cid create!')

    # 填充 new_df 的 new_container_id 列
    new_df['new_container_id'] = [
        cid_to_new_cid[cid]
        for cid in containeridList
    ] # ---new_container_id为None的，表明它对应的节点xxx上没有容器---
    
    containeridSeries = pd.Series(containeridList)
    # 填充 new_df 的 machine_id 列
    new_df['machine_id'] = np.where(
        containeridSeries.str.startswith('None'),  # 使用 Pandas 的 str 属性
        containeridSeries.str.split("+").str[1],
        containeridSeries.map(index_mapping)
    )
    
    # 按machine_id升序排列，重新编号
    new_df['machine_num'] = new_df['machine_id'].str.extract(r'NODE_(\d+)', expand=False).astype(int)
    new_df = new_df.sort_values(by='machine_num', ascending=True)
    new_df['new_machine_num']=new_df['machine_num'].rank(ascending=True, method='dense').astype(int)
    # print(f'new_df is \n{new_df}')

    # 把mapping转回x_t -------[有些key对应的value是'None']----------
    x_t0 = new_df.groupby('new_machine_num')['new_container_id'].apply(list).to_dict()
    # print(f'len of x_t0 is {len(x_t0)}')
    # 得到各个 container 的利用率【只获取那些对应node节点存在的container的利用率】
    cpu_t0 = sorted_df_filtered['cpu_util'].values
    mem_t0 = sorted_df_filtered['mem_util'].values

    # 计算 container 的 cpu 和 mem 利用率
    assert len(cpu_t0_node) == len(cpu_t0) and len(mem_t0_node) == len(mem_t0)
    cpu_t0 = cpu_t0 * cpu_t0_node * 100 + 1
    mem_t0 = mem_t0 * mem_t0_node * 100 + 1

    # 再对cpu_t0，mem_t0做归一化，得带百分比
    cpu_t0_max=max(cpu_t0)+1
    cpu_t0_min=min(cpu_t0)-1
    cpu_t0=MaxMinMethod(cpu_t0_max,cpu_t0_min,cpu_t0)*100

    mem_t0_max=max(mem_t0)+1
    mem_t0_min=min(mem_t0)-1
    mem_t0=MaxMinMethod(mem_t0_max,mem_t0_min,mem_t0)*100

    # print(f'after normalize, type of cpu_t0 is: {type(cpu_t0)}, cpu_t0 is: \n{cpu_t0}')
    # print(f'after normalize, type of mem_t0 is: {type(mem_t0)}, mem_t0 is: \n{mem_t0}')
    # print(f'after normalize, max cpu is {max(cpu_t0)}, min cpu is {min(cpu_t0)}, max mem is {max(mem_t0)}, min mem is {min(mem_t0)}')
    # sorted_df_filtered.set_index(['new_machine_num', 'new_container_id'])
    return new_df, cpu_t0, mem_t0, x_t0, trace

def MaxMinMethod(max, min, value):
    return (value-min)/(max-min)



# from utl import ResourceUsage1,isAllUnderLoad, recordTrace, recordInitTrace, getMapping_Pro, recordTrace_pro, isAllUnderLoad1
# [sorted_df, cpu_t0, mem_t0, x_t0]=getMappingAndUtil_pro('./data/processed_data/MSMetrics_1d.csv','./data/processed_data/NodeMetrics_1d.csv', 22) 
# print(cpu_t0)
# print(f'len of sorted_df is {len(sorted_df)}')
# sorted_df.to_csv('sorted_df_init.csv', index=False)

# res_CPU=ResourceUsage1(cpu_t0, x_t0)
# res_MEM=ResourceUsage1(mem_t0, x_t0)
# print(res_CPU)
# print(res_MEM)
# print(sum(res_CPU)/len(res_CPU))
# print(sum(res_MEM)/len(res_MEM))
# import json
# trace=recordInitTrace(sorted_df)
# json_str=json.dumps(trace, indent=0)
# with open('trace_init.txt', 'w') as f:
#     f.write(json_str)
#     f.write('\n')

# placement=Sandpiper_algo1(x_t0, cpu_t0, mem_t0, 75, 75)
# print(f'len of placement is {len(placement)}')
# json_str=json.dumps(placement, indent=0)
# with open('placement.txt', 'w') as f:
#     f.write(json_str)
#     f.write('\n')

# mapping=getMapping_Pro(placement, sorted_df)
# print(f'len of mapping is {len(mapping)}')
# json_str=json.dumps(mapping, indent=0)
# with open('mapping.txt', 'w') as f:
#     f.write(json_str)
#     f.write('\n')

# trace=recordTrace_pro(trace, mapping)
# print(f'len of trace is {len(trace)}')
# json_str=json.dumps(trace, indent=0)
# with open('trace.txt', 'w') as f:
#     f.write(json_str)
#     f.write('\n')

# print('1 times finish')
# [sorted_df, cpu_t0, mem_t0, x_t0, trace]=getMappingUtilForPlacement('./data/processed_data/MSMetrics_1d.csv','./data/processed_data/NodeMetrics_1d.csv', mapping, trace, 4)
# print(f'len of sorted_df is {len(sorted_df)}')
# sorted_df.to_csv('sorted_df_1.csv', index=False)
# json_str=json.dumps(x_t0, indent=0)
# with open('xt_1.txt', 'w') as f:
#     f.write(json_str)
#     f.write('\n')

# placement=Sandpiper_algo1(x_t0, cpu_t0, mem_t0, 75, 75)
# print(f'len of placement is {len(placement)}')
# json_str=json.dumps(placement, indent=0)
# with open('placement_1.txt', 'w') as f:
#     f.write(json_str)
#     f.write('\n')

# mapping=getMapping_Pro(placement, sorted_df)
# print(f'len of mapping is {len(mapping)}')
# json_str=json.dumps(mapping, indent=0)
# with open('mapping_1.txt', 'w') as f:
#     f.write(json_str)
#     f.write('\n')

# trace=recordTrace_pro(trace, mapping)
# print(f'len of trace is {len(trace)}')
# json_str=json.dumps(trace, indent=0)
# with open('trace_1.txt', 'w') as f:
#     f.write(json_str)
#     f.write('\n')

# print('2 times finish')
# [sorted_df, cpu_t0, mem_t0, x_t0, trace]=getMappingUtilForPlacement('./cnmap.csv','./nodeutil.csv', mapping, trace, 21*60*1000)
# print(f'len of sorted_df is {len(sorted_df)}')
# sorted_df.to_csv('sorted_df_2.csv', index=False)
# json_str=json.dumps(x_t0, indent=0)
# with open('xt_2.txt', 'w') as f:
#     f.write(json_str)
#     f.write('\n')

# placement=Sandpiper_algo1(x_t0, cpu_t0, mem_t0, 75, 75)
# print(f'len of placement is {len(placement)}')
# json_str=json.dumps(placement, indent=0)
# with open('placement_2.txt', 'w') as f:
#     f.write(json_str)
#     f.write('\n')

# mapping=getMapping_Pro(placement, sorted_df)
# print(f'len of mapping is {len(mapping)}')
# json_str=json.dumps(mapping, indent=0)
# with open('mapping_2.txt', 'w') as f:
#     f.write(json_str)
#     f.write('\n')

# trace=recordTrace_pro(trace, mapping)
# print(f'len of trace is {len(trace)}')
# json_str=json.dumps(trace, indent=0)
# with open('trace_2.txt', 'w') as f:
#     f.write(json_str)
#     f.write('\n')