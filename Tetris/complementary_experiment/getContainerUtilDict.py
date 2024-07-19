import pandas as pd
import json
import gc
import os
from collections import defaultdict

containerUtilDict = defaultdict(list)
container_path = './data/processed_data'
output_filename = 'containerUtilDict.txt'
chunksize = 100000  # 根据系统和文件大小调整每块大小

# 处理函数，处理每个CSV文件
def process_csv(filepath, containerUtilDict):
    print(f'Processing {filepath}')
    
    reader = pd.read_csv(filepath, chunksize=chunksize)
    print(f'read {filepath} finish')
    for chunk in reader:
        for idx, row in chunk.iterrows():
            cpu_util = row['cpu_util']
            mem_util = row['mem_util']
            cid = row['container_id']
            
            containerUtilDict[cid].append([cpu_util, mem_util])
    
    print(f'Completed processing {filepath}')
    

# 主处理循环，处理每个CSV文件
for i in range(1, 14):
    filepath = os.path.join(container_path, f'MSMetrics_{i}d.csv')
    process_csv(filepath, containerUtilDict)
    gc.collect()
print('Processing completed. Writing to file...')

# 转换为普通字典
containerUtilDict = dict(containerUtilDict)

# 写入JSON文件
with open(output_filename, 'w') as file:
    json.dump(containerUtilDict, file, indent=4)

print(f'Write to file {output_filename} successful')

