import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_sxy=pd.read_csv('activetime_sxy.csv')
uptime_sxy=df_sxy.iloc[:, 1].tolist()
uptime_sxy=[x/348 for x in uptime_sxy]

uptime_wzq={}
with open('containerUptime_13d.txt', 'r') as file:
    jsonstr=file.read()
    uptime_dict=json.loads(jsonstr)
    uptime_wzq=list(uptime_dict.values())
    uptime_wzq=[x/60 for x in uptime_wzq]

# 计算CDF
sorted_data1 = np.sort(uptime_sxy)
sorted_data2 = np.sort(uptime_wzq)
cdf1 = 100*np.arange(1, len(sorted_data1) + 1) / len(sorted_data1)
cdf2 = 100*np.arange(1, len(sorted_data2) + 1) / len(sorted_data2)

plt.figure(figsize=(12,8))
plt.xlabel('Startup time (hours)', fontsize = 40)
plt.ylabel('CDF (%) of containers', fontsize = 40)
ax = plt.gca()
ax.tick_params(width = 4, length = 8, labelsize = 40)
bwith = 4
ax.spines['top'].set_linewidth(bwith)
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

xticks=[0,50,100,150,200,250,300,350]
yticks=[0,20,40,60,80,100]
ax.set_xticks(xticks)
ax.set_yticks(yticks)
# 绘制CDF曲线
plt.plot(sorted_data1, cdf1, linewidth = 4.0,label='trace v2018')
plt.plot(sorted_data2, cdf2, linewidth = 4.0,label='trace v2022')

plt.grid(axis='y')

# 添加每个Y值处的水平线
# for y in range(0,101,20):
#     plt.axhline(y, color='gray', linestyle='-', alpha=0.5)
# plt.grid(True)
# 添加图例
plt.rcParams.update({'font.size':35})
plt.legend(loc='upper left')

plt.savefig('uptime_cdf_compare.png',bbox_inches = 'tight', transparent = True) 
plt.close()



