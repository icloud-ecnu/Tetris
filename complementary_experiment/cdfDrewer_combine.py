import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 把第一天的和后12天的结合
def plot_cdf(data):
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
    
    plt.plot(sorted_data, yvals)
    plt.xlabel('Values')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function')
    plt.grid(True)
    plt.savefig('uptime_cdf.png') 
    plt.close()

uptime_dict_12={}
with open('containerUptime.txt', 'r') as file:
    jsonstr=file.read()
    uptime_dict_12=json.loads(jsonstr)

uptime_dict_1={}
with open('containerUptime_1.txt', 'r') as file:
    jsonstr=file.read()
    uptime_dict_1=json.loads(jsonstr)

# 把第一天的数据和后12天的数据拼合起来
for cid, minute in uptime_dict_1.items():
    if cid not in uptime_dict_12:
        uptime_dict_12[cid]=minute
    else:
        uptime_dict_12[cid]+=minute

# jsonstr=json.dumps(uptime_dict_12, indent=0)
# with open('containerUptime_13d.txt', 'w') as f:
#     f.write(jsonstr)
#     f.write('\n')
uptime_list=list(uptime_dict_12.values())
uptime_list=[x/60 for x in uptime_list]
plot_cdf(uptime_list)