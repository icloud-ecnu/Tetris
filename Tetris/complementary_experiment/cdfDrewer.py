import json
import matplotlib.pyplot as plt
import numpy as np


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


with open('containerUptime.txt', 'r') as file:
    jsonstr=file.read()
    uptime_dict=json.loads(jsonstr)
    uptime_list=list(uptime_dict.values())
    plot_cdf(uptime_list)