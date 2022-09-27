from time import time
import csv
import os

class Scheduler(object):
    def __init__(self, env,algorithm,metricFile,motivationFile):
        self.env = env
        self.simulation = None
        self.cluster = None
        self.algorithm = algorithm
        self.dir = os.getcwd()
        self.motivationFile = motivationFile
        
        with open(motivationFile,"w") as f:
                writer = csv.writer(f)
                writer.writerow(['time','container','metric'])
        with open(self.metrifile,'w') as f:
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
            value,eval_bal,eval_mig = algorithm(self.cluster, self.env.now,self.motivationFile)
            sums += value
            after = time()-start
            alltime+=after
            vms = [ len(v) for k,v in self.cluster.isAllUnderLoad(self.env.now,self.sand ).items()]
            vmlen = sum(vms)
            
            with open(self.metrifile,'a') as f:
                writer = csv.writer(f)
                writer.writerow([end,eval_bal,eval_mig,value,sums,after,alltime,vmlen])
            yield self.env.timeout(1)
        
        print('now finish time:', self.env.now)


    