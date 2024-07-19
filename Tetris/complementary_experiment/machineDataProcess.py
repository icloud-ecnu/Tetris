import pandas as pd
import os
import numpy as np
import csv

basepath = './data/NodeMetrics'  # 原始数据路径
basename = 'NodeMetricsUpdate'  # 原始数据文件名前缀
output_dir = './data/processed_data'  # 输出数据路径

# 如果输出路径不存在，则创建
os.makedirs(output_dir, exist_ok=True)

# 处理13天数据，每小时处理一次
for i in range(2, 2*13, 2):
    print(f'i={i}')
    day=int(i/2)+1

    output_file = os.path.join(output_dir, f'NodeMetrics_{day}d.csv')
    with open(output_file, 'w', newline='') as file:
        print(f'{output_file} 创建成功！')
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'machine_id', 'cpu_util', 'mem_util'])
    
    # 逐块读取当前半天和后半天的数据
    chunksize = 100000  # 每次读取100万行数据
    chunks = []

    print('开始读取数据')
    for j in range(i, i + 2):
        filename = os.path.join(basepath, f'{basename}_{j}.csv')
        for chunk in pd.read_csv(filename, chunksize=chunksize):
            chunks.append(chunk)
    print(f'{day}d 的数据读取完成')

    # 合并两个DataFrame
    df = pd.concat(chunks, ignore_index=True)
    del chunks

    # 对于一天内的24小时，获取每一个小时的数据并处理
    starthour=24*(day-1)
    for hour in range(starthour, starthour+24):
        start_timestamp = hour * 3600 * 1000
        end_timestamp = (hour + 1) * 3600 * 1000
        # 筛选当前小时的数据
        mask = (df['timestamp'] >= start_timestamp) & (df['timestamp'] < end_timestamp)
        df_hour = df.loc[mask]
        if not df_hour.empty:
            # 按节点ID分组，计算平均cpu_utilization和memory_utilization
            grouped_df_hour = df.groupby('nodeid').agg({
                'cpu_utilization': 'mean',
                'memory_utilization': 'mean'
            }).reset_index()
        
            # 将结果写入每小时的CSV文件
            with open(output_file, 'a', newline='') as file:
                writer = csv.writer(file)
                for index, row in grouped_df_hour.iterrows():
                    writer.writerow([hour, row['nodeid'], row['cpu_utilization'], row['memory_utilization']])
        
    # 释放内存资源
    del df
    
    print(f'{day}d 的数据处理完成')

print('所有处理完成！')