
import numpy as np
from sandpiper import Sandpiper_algo
from abc import ABC, abstractmethod
from sxyAlgo.Algorithm_sxy import Algorithm_sxy
from utl import CostOfLoadBalance, CostOfMigration
import csv


class Algorithm(ABC):
    @abstractmethod
    def __call__(self, *args):
        pass


class Scheduler():
    def __init__(self) -> None:
        self.sxy_algo = Algorithm_sxy()
        self.Filename = None
        pass

    def __call__(self, nodes, cpudict, memdict, algo, cluster=None, t=None):

        podnum = len(cpudict)
        nodenum = len(nodes)
        print("podnum = {} nodenum = {}".format(podnum, nodenum))

        if algo == "sandpiper":
            self.Filename = './metric/sandpiper.csv'
            self.x_t0 = np.zeros([podnum, nodenum])
            print("podnum and nodenum", podnum, nodenum)

            self.cpu_t0, self.mem_t0 = np.zeros(podnum), np.zeros(podnum)
            self.getMatrix(nodes, cpudict, memdict)
            placement = Sandpiper_algo(
                self.x_t0, self.cpu_t0, self.mem_t0, CPU_MAX=13, MEM_MAX=100)
            eval_bal = CostOfLoadBalance(
                self.cpu_t0, self.mem_t0, placement, b=0.0025)
            eval_mig = CostOfMigration(self.x_t0, placement, self.mem_t0)
            value = eval_bal+(podnum-1)*eval_mig*0.004
        elif algo == "sxy":

            print("sxy")
            self.Filename = './metric/sxy.csv'
            value, eval_bal, eval_mig = self.sxy_algo(cluster, t)
        elif algo == "drl":
            self.Filename = './metric/drl.csv'

            print("drl")
        with open(self.Filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([value, eval_bal, eval_mig])
        if algo == "sxy":
            return self.sxyDict(cluster), eval_mig
        return self.MatrixToDict(placement, nodenum), eval_mig

    def sxyDict(self, cluster):
        nodes = cluster.nodes
        newnodes = {}
        nodename_prefix = "k8s-node"
        for nodeNum, v in nodes.items():
            newnodes[nodename_prefix +
                     str(nodeNum+1)] = set([e.name for e in v.containers.values()])
        return newnodes

    def MatrixToDict(self, placement, nodenum):

        nodes = {}
        nodename_prefix = "k8s-node"
        for nodeNum in range(1, nodenum+1):
            x = np.where(placement[:, nodeNum-1] == 1)[0]

            nodes[nodename_prefix +
                  str(nodeNum)] = set([self.idxTopodname[e] for e in x])
        return nodes

    def getMatrix(self, nodes: dict, cpudict: dict, memdict: dict):

        x_t0 = self.x_t0
        for nodename, podnameSet in nodes.items():

            i = int(nodename[8:])-1
            for podname in podnameSet:

                j = self.podnameToidx[podname]

                x_t0[j][i] = 1
        cpu_t0, mem_t0 = self.cpu_t0, self.mem_t0
        for podenameCpu, podenameMem in zip(cpudict.keys(), memdict.keys()):
            cpu_t0[self.podnameToidx[podenameCpu]] = cpudict[podenameCpu][-1]
            mem_t0[self.podnameToidx[podenameMem]] = memdict[podenameMem][-1]

    def addChange(self, podnameToidx, idxTopodname):
        self.podnameToidx, self.idxTopodname = podnameToidx, idxTopodname
