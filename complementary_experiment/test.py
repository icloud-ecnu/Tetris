import numpy as np
import os
import pandas as pd
import json

# df=pd.read_csv('data/NodeMetrics/NodeMetricsUpdate_2.csv')

# df=df[df['timestamp']==(26+0)*60*1000]
# print(len(df))
# uptime_dict={}
# basepath='./data/MSMetrics'
# for idx, file in enumerate(os.listdir(basepath)):
#     filename=os.path.join(basepath, file)
#     print(f'read file {file} finish')
#     df=pd.read_csv(filename)
#     df=df.drop_duplicates(subset=['timestamp', 'msinstanceid'],keep='first')
#     for cid in df['msinstanceid'].values:
#         if cid not in uptime_dict:
#             uptime_dict[cid]=1
#         else:
#             uptime_dict[cid]+=1
#     print(f'record file {file} finish!')
# json_str=json.dumps(uptime_dict, indent=0)
# print('get json_str')
# with open('containerUptime_1.txt', 'w') as f:
#     f.write(json_str)
#     f.write('\n')

# mapping={'a':['apple', 'alice'], 'b':[], 'c':['canda', 'candy']}
# index_mapping = {c: k for k, v in mapping.items() for c in v}
# print(index_mapping)

# with open('trace_14h.txt', 'r') as f:
#     str=f.read()
#     trace=json.loads(str)
#     longdict={k:v for k,v in trace.items() if len(v)>=3}
#     print(longdict)
#     print(len(longdict.keys()))

# SSDP_ac=np.array([71.04, 78.72, 78.22, 83.76, 85.47, 93.24, 97.06, 97.28, 95.96, 95.24])
# SDP_ac=np.array([69.05, 73.85, 77.50, 75.51, 78.63, 90.24, 91.71, 95.21, 89.42, 89.60])
# SSDP_f1=np.array([0.70,0.75,0.761,0.7900,0.8268,0.9011,0.9458,0.9316,0.9273,0.9331])
# SDP_f1=np.array([0.66,0.72,0.74,0.7460,0.7720,0.8909,0.9016,0.9234,0.8692,0.8712])

# improve_ac=100*(SSDP_ac-SDP_ac)/SDP_ac
# improve_f1=100*(SSDP_f1-SDP_f1)/SDP_f1
# print(f'ac 提升率为{improve_ac}，最大值为{max(improve_ac)}， 最小值为{min(improve_ac)}')
# print(f'f1 提升率为{improve_f1}，最大值为{max(improve_f1)}， 最小值为{min(improve_f1)}')

# import ijson
# count=0
# with open('/mnt/d/school/tetris/containerUtilDict.txt', 'r') as file:
#     parser=ijson.kvitems(file, '')
#     for cid, util_data in parser:
#         count+=1
# print(f'count of container is {count}')

# nonperiodic_cpu=set(pd.read_csv('./wzq/nonperiodic_cpu.csv')['id'].values.tolist())
# periodic_cpu=set(pd.read_csv('./wzq/periodic_cpu.csv')['id'].values.tolist())
# nonperiodic_mem=set(pd.read_csv('./wzq/nonperiodic_mem.csv')['id'].values.tolist())
# periodic_mem=set(pd.read_csv('./wzq/periodic_mem.csv')['id'].values.tolist())

# periodic_inc=periodic_cpu & periodic_mem
# nonperiodic_inc=nonperiodic_cpu & nonperiodic_mem

# print(list(periodic_inc)[:10])
# print(list(nonperiodic_inc)[:10])

# # MS_49499_POD_404 周期
# # MS_5229_POD_22 非周期
# p_data=[]
# np_data=[]
# flag=0
# import ijson
# with open('/mnt/d/school/tetris/containerUtilDict.txt', 'r') as file:
#     parser=ijson.kvitems(file, '')
#     print('get parser')
#     for cid, util_data in parser:
#         if flag==2:
#             break
#         if cid == 'MS_18426_POD_108':
#             p_data=util_data
#             flag+=1
#         if cid == 'MS_13385_POD_0':
#             np_data=util_data
#             flag+=1
# print('get all')
# df=pd.DataFrame({
#     'cpu': [util[0] for util in p_data],
#     'mem': [util[1] for util in p_data]})

# df.to_csv('instance_MS_18426_POD_108.csv', index=False)

# df=pd.DataFrame({
#     'cpu': [util[0] for util in np_data],
#     'mem': [util[1] for util in np_data]})

# df.to_csv('instance_MS_13385_POD_0.csv', index=False)
filepath='/mnt/d/school/tetris/simulation'
filepath=os.path.join(filepath, '10-metric_w6z20.csv')
df=pd.read_csv(filepath)
eval_bal=df['eval_bal'].values
eval_mig=df['eval_mig'].values
print(f'平均负载均衡开销：{np.mean(eval_bal[:24])}')
print(f'平均迁移开销：{np.mean(eval_mig[:24])}')

