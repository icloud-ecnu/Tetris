import simpy
import numpy as np
import sys
sys.path.append('..')
from cluster import Cluster
from scheduler import Scheduler

import profile


class Simulation(object):
    def __init__(self, configs,algorithm,metricFile,movtivationFile,args=None):
        instance_configs,machine_configs,mac_ids,inc_ids= configs
        self.sand = False
        
        if args is not None:
            self.drl = args.drl
            self.sand = args.sandpiper
        self.env = simpy.Environment()
        self.cluster = Cluster()
        self.cluster.add_old_new(mac_ids,inc_ids)
        self.cluster.configure_machines(machine_configs)
        self.cluster.configure_instances(instance_configs)
        
        self.scheduler = Scheduler(self.env,algorithm,self.sand,metricFile,movtivationFile)

        self.scheduler.attach(self)

    
    def run(self):
        
        self.env.process(self.scheduler.run())
        print("simulation")
        self.env.run()
    
    
    def finished(self,clock):
        if clock>=len(self.cluster.instances[0].cpulist)-2:
            return True
        
        return False
 