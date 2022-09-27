import pandas as pd
import argparse
import json
import os
from algorithm import Algorithm_sxy
from simulation import Simulation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--arima', type=bool, default=False)
    args = parser.parse_args()
    filepath = './data/resource_1h/'
    # for the data in 10 seconds
    # filepath = './data/resource_10s/'

    # metric_log_filename = os.path.join(os.getcwd(), 'metric')
    metric_log_filename = 'metric'

    from data.loader_reduce import InstanceConfigLoader
    
    test_array = [[3989, 67437], [100, 1700], [500, 8500],
                  [1000, 17000], [2000, 34000], [3000, 51000]]
    configs = InstanceConfigLoader(filepath, test_array)
    algorithm = Algorithm_sxy()
    
    for i, config in enumerate(configs):

        res_struct_filename = os.path.join(
            os.getcwd(), str(len(config[1]))+'-struct.json')
        metricFile = os.path.join(
            os.getcwd(), str(len(config[1]))+'-metric.csv')
        movtivationFile = os.path.join(
            os.getcwd(), str(len(config[1]))+'-motivation.csv')

        print("####################################",
              i, "#############################")
        
        sim = Simulation(config, algorithm, metricFile, movtivationFile, args)
        sim.run()
