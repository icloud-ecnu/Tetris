from instance import Instance
import numpy as np


class MachineConfig(object):
    def __init__(self, machine_id, cpu_capacity, memory_capacity):
        self.id = machine_id
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity


class Machine(object):
    def __init__(self, machine_config):
        self.id = machine_config.id
        self.cpu_capacity = machine_config.cpu_capacity
        self.mem_capacity = machine_config.memory_capacity
        self.isMigration = False
        self.cluster = None
        self.instances_bak = None
        self.instances = {}
        self.len = 0

        self.cpuPluPredict = None
        self.memPluPredict = None

        self.nowPluPredictCost_w = {}

        self.cpu_sum_w = None
        self.mem_sum_w = None

        self.CsPluMs = None
        self.CsPluMs_migraton = None

    def attach(self, cluster):
        self.cluster = cluster

    
    def add_instance(self, instance_config):

        instance = Instance(instance_config)
        instance_config.machine_id = self.id
        instance.attach(self)
        self.instances[instance.id] = instance

    
    def pop(self, instance_id):
        instance = self.instances.pop(instance_id)
        instance.machine = None
        return instance

    
    def push(self, instance):
        self.instances[instance.id] = instance
        instance.attach(self)

    
    def getEveryTimeCpuList(self, clock, w):
        nextclock = clock+1
        instances = self.instances
        self.len = len(instances)
        self.cpuPluPredict = {v.id: np.array(
            v.cpulist[clock:nextclock]+v.predict(clock, w)[clock]["cpu"]) for v in instances.values()}
        self.memPluPredict = {v.id: np.array(
            v.memlist[clock:nextclock]+v.predicts[clock]["mem"]) for v in instances.values() for v in instances.values()}

    
    def cupSumAndMemSum(self):
        self.cpu_sum_w = np.sum(
            np.array([v for k, v in self.cpuPluPredict.items()]), axis=0)
        self.mem_sum_w = np.sum(
            np.array([v for k, v in self.memPluPredict.items()]), axis=0)

    
    def cost_first(self, clock, w, b=0.0025):
        self.getEveryTimeCpuList(clock, w)
        self.cupSumAndMemSum()

        csplums = []
        pm_cost = 0

        cpu_vm = np.array([v for k, v in self.cpuPluPredict.items()])
        mem_vm = np.array([v for k, v in self.memPluPredict.items()])
        csplums = [0.0 for i in range(w)]
        pm_cost = 0
        
        for i in range(len(cpu_vm)-1):
            for t in range(w):
                c = cpu_vm[i][t] * np.sum(cpu_vm[i+1:, t])
                m = mem_vm[i][t] * np.sum(mem_vm[i+1:, t])
                csplums[t] += c + b * m
        
        for t in range(w):
            if csplums[t] > 0.5:
                csplums[t] -= 0.5
        pm_cost = np.sum(csplums)

        self.CsPluMs = csplums
        self.nowPluPredictCost_w[clock] = pm_cost

        return csplums[0]

    
    def getnowPluPredictCost(self, clock, w, b):
        if clock not in self.nowPluPredictCost_w or self.nowPluPredictCost_w[clock] is None:
            self.cost_first(clock, w, b)
        return self.nowPluPredictCost_w[clock]

    
    def canAddorNot(self, cpu, mem, t):
        return self.cpu_sum[t]+cpu < self.cpu_capacity \
            and self.mem_sum[t] + mem < self.mem_capacity

    
    def migrateOut(self, vmid, t):
        self.pop(vmid)
        return self.memPluPredict[vmid][t]*2

    
    def migrateIn(self, vmid, t):
        self.push(vmid)

    
    def afterMigration_cost(self, clock, t, w, b):

        cost_t = {}
        cpu_vm = np.array([v for k, v in self.cpuPluPredict.items()])
        mem_vm = np.array([v for k, v in self.memPluPredict.items()])

        csplums = [0.0 for i in range(w)]
        
        for i in range(len(cpu_vm)-1):
            for t in range(w):
                c = cpu_vm[i][t] * np.sum(cpu_vm[i+1:, t])
                m = mem_vm[i][t] * np.sum(mem_vm[i+1:, t])
                csplums[t] += c + b * m

        cost_t[t] = csplums[t]

        self.CsPluMs_migraton = csplums
        return cost_t[t]

    
    def afterOneContainerMigration(self, clock, w, b):
        self.getEveryTimeCpuList(clock, w)
        self.cupSumAndMemSum()
        csplums = []
        pm_cost = 0
        cpu_vm = np.array([v for k, v in self.cpuPluPredict.items()])
        mem_vm = np.array([v for k, v in self.memPluPredict.items()])
        csplums = [0.0 for i in range(w)]
        pm_cost = 0
        
        for i in range(len(cpu_vm)-1):
            for t in range(w):
                c = cpu_vm[i][t] * np.sum(cpu_vm[i+1:, t])
                m = mem_vm[i][t] * np.sum(mem_vm[i+1:, t])
                csplums[t] += c + b * m
        
        for t in range(w):
            if csplums[t] > 0.5:
                csplums[t] -= 0.5
        
        pm_cost = np.sum(csplums)
        
        return pm_cost

    
    @property
    def cpu(self):
        occupied = 0
        
        for instance in self.instances.values():
            occupied += instance.cpu
        
        return self.cpu_capacity - occupied

    
    @property
    def mem(self):
        occupied = 0
        
        for instance in self.instances.values():
            occupied += instance.mem

        return self.mem_capacity - occupied

    
    @property
    def cpusum(self):
        occupied = 0
        
        for instance in self.instances.values():
            occupied += instance.cpu
        
        return self.cpu_capacity - occupied

    
    @property
    def memsum(self):
        occupied = 0
        
        for instance in self.instances.values():
            occupied += instance.mem

        return self.mem_capacity - occupied