import pandas as pd
import os
import numpy as np
import csv

basepath = 'data/MSMetrics'  # 原始数据路径
basename = 'MSMetricsUpdate'  # 原始数据文件名前缀
output_dir = './data/processed_data'  # 输出数据路径

# 如果输出路径不存在，则创建
os.makedirs(output_dir, exist_ok=True)

# 处理13天数据，每小时处理一次
day = 1
for i in range(0, 48, 2):
    print(f'i={i}')
    if i % 48 == 0:
        output_file = os.path.join(output_dir, f'MSMetrics{day}.csv')
        with open(output_file, 'w', newline='') as file:
            print(f'{output_file} 创建成功！')
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'container_id', 'machine_id', 'cpu_util', 'mem_util'])
        day += 1
    
    # 逐块读取当前小时和下半小时的数据
    chunksize = 1000000  # 每次读取100万行数据
    chunks = []
    print('开始读取数据')
    # 读取当前小时的数据
    for chunk in pd.read_csv(os.path.join(basepath, f'{basename}_{i}.csv'), chunksize=chunksize):
        chunks.append(chunk)
    
    # 读取下半小时的数据
    for chunk in pd.read_csv(os.path.join(basepath, f'{basename}_{i+1}.csv'), chunksize=chunksize):
        chunks.append(chunk)
    print(f'{int(i / 2)}h 的数据读取完成')

    # 合并两个DataFrame
    df = pd.concat(chunks, ignore_index=True)
    
    # 按容器ID分组，计算平均cpu_utilization和memory_utilization
    grouped_df = df.groupby('msinstanceid').agg({
        'nodeid': 'first',
        'cpu_utilization': 'mean',
        'memory_utilization': 'mean'
    }).reset_index()
    
    # 将结果写入每小时的CSV文件
    with open(output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        for index, row in grouped_df.iterrows():
            writer.writerow([int(i / 2), row['msinstanceid'], row['nodeid'], row['cpu_utilization'], row['memory_utilization']])
        
    # 释放内存资源
    del chunks
    del df
    
    print(f'{int(i / 2)}h 的数据处理完成')

print('所有处理完成！')


# import pandas as pd
# import os
# import numpy as np
# import csv

# basepath='data/MSMetrics'
# basename='MSMetricsUpdate'
# day=1
# # 以1h为单位处理13d的数据，处理方式为平均值
# # 计划把一天的数据写为一个csv，包含24h的数据
# for i in range(0,48*13,2):
#     if i % 48 ==0:
#             file=open('./data/processed_data/MSMetrics'+str(day)+'.csv', 'w', newline='')
#             print(f'{file.name} create!')
#             writer=csv.writer(file)
#             writer.writerow(['timestamp','container_id','machine_id','cpu_util','mem_util']) 
#             day+=1

#     df1=pd.read_csv(os.path.join(basepath, basename+'_'+str(i)+'.csv')) # 前半小时
#     df2=pd.read_csv(os.path.join(basepath, basename+'_'+str(i+1)+'.csv')) # 后半小时
#     # 获取这一小时中出现过的所有容器id
#     existContainers=set(df1['msinstanceid'].values.tolist()+df2['msinstanceid'].values.tolist())
#     print(f'get all container ids in {int(i/2)}\'s hour, start processing!')
#     for cid in existContainers:
#         # 从两个df中获取一小时内cid的cpu、mem利用率
#         df1_filtered=df1[df1['msinstanceid']==cid]
#         df2_filtered=df2[df2['msinstanceid']==cid]
#         cpu_in_hour=np.concatenate((df1_filtered['cpu_utilization'].values,df2_filtered['cpu_utilization'].values))  
#         mem_in_hour=np.concatenate((df1_filtered['memory_utilization'].values,df2_filtered['memory_utilization'].values)) 
#         # 计算均值
#         cpu=np.mean(cpu_in_hour)
#         mem=np.mean(mem_in_hour)
#         # 写入csv文件
#         timestamp=i/2 
#         mid=df2_filtered['nodeid'].iloc[0] if df1_filtered.empty else df1_filtered['nodeid'].iloc[0]
#         writer.writerow([int(timestamp),cid,mid,cpu,mem])
#     print('{int(i/2)}\'s hour done')

# file.close()
# print('all process done!')

        