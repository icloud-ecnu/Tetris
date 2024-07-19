from time import time
import csv
import os

class Scheduler(object):
    def __init__(self,env,algorithm,sand,metricFile,motivationFile):
        self.env = env
        self.simulation = None
        self.cluster = None
        self.algorithm = algorithm
        self.dir = os.getcwd()
        self.motivationFile = motivationFile
        self.metricFile=metricFile
        self.sand = sand
        
        with open(motivationFile,"w") as f:
                writer = csv.writer(f)
                writer.writerow(['time','container','metric'])
        with open(self.metricFile,'w') as f:
            writer = csv.writer(f)
            writer.writerow(['clock','eval_bal','eval_mig','sum','sums','time','total_time','violation'])
        
    
    def attach(self, simulation):
        self.simulation = simulation
        self.cluster = simulation.cluster
    
    
    def run(self):
        sums = 0
        alltime = 0
        algorithm = self.algorithm
        
        while not self.simulation.finished(self.env.now):
            start = time()
            end = self.env.now

            algo_start = time()
            value,eval_bal,eval_mig = algorithm(self.cluster, self.env.now,self.motivationFile)
            algo_end = time() - algo_start
            print(f"Algorithm execution time: {algo_end:.2f}s")
            sums += value
            after = time()-start
            alltime+=after

            load_start = time()
            vms = [ len(v) for k,v in self.cluster.isAllUnderLoad(self.env.now,self.sand).items()]
            load_end = time() - load_start
            print(f"isAllUnderLoad execution time: {load_end:.2f}s")
            vmlen = sum(vms)
            
            with open(self.metricFile,'a') as f:
                writer = csv.writer(f)
                writer.writerow([end,eval_bal,eval_mig,value,sums,after,alltime,vmlen])
            yield self.env.timeout(1)
        
        print('now finish time:', self.env.now)


    