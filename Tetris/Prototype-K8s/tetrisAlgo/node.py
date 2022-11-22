
from sxyAlgo.container import Container
import numpy as np


class Node(object):
    def __init__(self, node_config):
        self.id = node_config["id"]
        self.nodename = node_config["nodeName"]
        self.cpu_capacity = node_config["cpu_capacity"]
        self.mem_capacity = node_config["mem_capacity"]
        self.isMigration = False
        self.cluster = None
        self.containers_bak = None
        self.containers = {}
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

    def add_container(self, container_config):

        container = Container(container_config)
        container_config.node_id = self.id
        container.attach(self)
        self.containers[container.id] = container

    def pop(self, container_id):
        container = self.containers.pop(container_id)
        container.node = None
        return container

    def push(self, container):
        self.containers[container.id] = container
        container.attach(self)

    def getEveryTimeCpuList(self, clock, w):

        nextclock = clock+1
        containers = self.containers
        self.len = len(containers)

        self.cpuPluPredict = {v.id: np.array(
            v.cpulist[clock:nextclock]+v.predict(clock, w)[clock]["cpu"]) for v in containers.values()}
        self.memPluPredict = {v.id: np.array(
            v.memlist[clock:nextclock]+v.predicts[clock]["mem"]) for v in containers.values() for v in containers.values()}

    def cupSumAndMemSum(self):

        self.cpu_sum_w = np.sum(
            np.array([v for k, v in self.cpuPluPredict.items()]), axis=0)
        self.mem_sum_w = np.sum(
            np.array([v for k, v in self.memPluPredict.items()]), axis=0)

    def cost_first(self, clock, w, b):
        self.getEveryTimeCpuList(clock, w)
        self.cupSumAndMemSum()
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

    @property
    def cpu(self):
        occupied = 0
        for container in self.containers.values():
            occupied += container.cpu
        return self.cpu_capacity - occupied

    @property
    def mem(self):
        occupied = 0
        for container in self.containers.values():
            occupied += container.mem

        return self.mem_capacity - occupied
