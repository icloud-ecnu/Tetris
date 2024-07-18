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

    # 将机器对象与集群（Cluster）关联起来
    def attach(self, cluster):
        self.cluster = cluster

    # 向机器中添加实例（Instance）
    def add_instance(self, instance_config):

        instance = Instance(instance_config)
        instance_config.machine_id = self.id
        instance.attach(self)
        self.instances[instance.id] = instance

    # 从机器中移除指定id的实例，并返回
    def pop(self, instance_id):
        instance = self.instances.pop(instance_id)
        instance.machine = None
        return instance

    # 将实例对象添加到机器中
    def push(self, instance):
        self.instances[instance.id] = instance
        instance.attach(self)

    # 根据时间窗口w大小，获取机器中每个实例的CPU使用情况列表
    def getEveryTimeCpuList(self, clock, w):
        nextclock = clock+1
        instances = self.instances
        self.len = len(instances)
        self.cpuPluPredict = {v.id: np.array(
            v.cpulist[clock:nextclock]+v.predict(clock, w)[clock]["cpu"]) for v in instances.values()}
        self.memPluPredict = {v.id: np.array(
            v.memlist[clock:nextclock]+v.predicts[clock]["mem"]) for v in instances.values() for v in instances.values()}

    # 计算机器中所有实例的CPU使用总和和内存使用总和
    def cupSumAndMemSum(self):
        self.cpu_sum_w = np.sum(
            np.array([v for k, v in self.cpuPluPredict.items()]), axis=0)
        self.mem_sum_w = np.sum(
            np.array([v for k, v in self.memPluPredict.items()]), axis=0)

    # 计算机器在指定时间范围内的成本
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

    # 获取指定时间点的成本值
    def getnowPluPredictCost(self, clock, w, b):
        if clock not in self.nowPluPredictCost_w or self.nowPluPredictCost_w[clock] is None:
            self.cost_first(clock, w, b)
        return self.nowPluPredictCost_w[clock]

    # 检查是否可以在指定时间点添加指定CPU和内存容量的实例到机器中
    def canAddorNot(self, cpu, mem, t):
        return self.cpu_sum[t]+cpu < self.cpu_capacity \
            and self.mem_sum[t] + mem < self.mem_capacity

    # 将指定id的实例从机器中迁出，返回迁出操作内存消耗
    def migrateOut(self, vmid, t):
        self.pop(vmid)
        return self.memPluPredict[vmid][t]*2

    # 将指定id的实例迁入到机器中
    def migrateIn(self, vmid, t):
        self.push(vmid)

    # 计算迁移后指定时间点的成本
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

    # 计算执行一次容器迁移后的成本
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

    # 返回机器中剩余的CPU容量
    @property
    def cpu(self):
        occupied = 0
        
        for instance in self.instances.values():
            occupied += instance.cpu
        
        return self.cpu_capacity - occupied

    # 返回机器中剩余的内存容量
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