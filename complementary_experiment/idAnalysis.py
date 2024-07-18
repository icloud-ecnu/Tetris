import pandas as pd
import os
import json

basepath='./data/MSMetrics'
basedf=pd.read_csv('./data/MSMetrics/MSMetricsUpdate_0.csv')
basedf=basedf.drop_duplicates(subset=['timestamp', 'msinstanceid'],keep='first')
cids=basedf[basedf['timestamp']==900000]['msinstanceid'].values.tolist()
cmidMap={k:list() for k in cids}
print('cmidMap create!')
for idx, file in enumerate(os.listdir(basepath)):
    filename=os.path.join(basepath, file)
    df_chunks=pd.read_csv(filename, iterator=True, chunksize=100000)
    print(f'read file {file} finish')
    for df in df_chunks:
        df=df.drop_duplicates(subset=['timestamp', 'msinstanceid'],keep='first')
        for idx, row in df.iterrows():
            cid=row['msinstanceid']
            mid=row['nodeid']
            if cid in cmidMap:
                if cmidMap[cid]==[]:
                    cmidMap[cid]=[mid]
                elif cmidMap[cid][-1]==mid:
                    continue
                else:
                    cmidMap[cid].append(mid)
    print(f'record file {file} finish!')

json_str=json.dumps(cmidMap, indent=0)
print('get json_str')
with open('cmidMap_1d.txt', 'w') as f:
    f.write(json_str)
    f.write('\n')