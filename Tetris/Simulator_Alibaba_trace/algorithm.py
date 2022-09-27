import numpy as np
from cluster import Cluster
from time import time
from pyDOE import lhs
from abc import ABC, abstractmethod
import csv


class Algorithm(ABC):
    @abstractmethod
    def __call__(self, *args):
        pass


class Algorithm_sxy(Algorithm):
    def __call__(self, cluster: Cluster, now, motivationFile):
        self.motivationFile = motivationFile
        self.cluster = cluster
        self.params = {"w": 3, "z": range(20), "k": 5, "u": 0.8, "v": 0.4, "a": 0.004,
                       "b": 0.0025, "y": 0.25, "N": len(cluster.instances),
                       "M": len(cluster.machines)}
        
        if now == 0:
            with open(motivationFile, "w") as f:
                writer = csv.writer(f)
                writer.writerow(['time', 'container', 'metric'])
        value, eval_bal, eval_mig = self.schedule(now)
        
        return value, eval_bal, eval_mig
    
    
    # @profile
    def schedule(self, now):
        start = time()
        params = self.params
        min_z, eval_bal, eval_mig, value = self.SchedulePolicy(params["z"], params["k"], params["w"], params["v"],
                                                               params["M"], params["a"], params["b"], params["y"], now)
        after = time()
        
        if min_z != -1:
            print("at ", now, "花费了", after-start, "s metric=",
                  min_z, eval_bal, eval_mig, value)
        else:
            print("at ", now, "没有最优，总共花费了", after -
                  start, eval_bal, eval_mig, value)
        
        return value, eval_bal, eval_mig

    
    def SchedulePolicy(self, Z, K, W, v, M, a, b, y, now):
        cluster = self.cluster
        lenx = len(cluster.instances[0].cpulist)
        s = time()
        cost_min, balfirst = cluster.cost_all_pm_first(now, W, b)
        print("算法开始执行 计算 cost_min    消耗了 %.2f s, cost_min = %.3f " %
              (time()-s, cost_min))
        
        s = time()
        over, under = self.findOverAndUnder(cluster, b, y, 0)
        print("算法开始执行 计算over & under 消耗了 %.2f s, length of over = %d  length of under = %d " % (
            time()-s, len(over), len(under)))
        CPU_t, MEM_t = None, None

        candidate_copy = {}
        min_z, balf, migf, valuef = -1, balfirst, 0, balfirst
        
        for z in Z:
            cost = 0
            findOV = (over, under)
            bal, mig, value = 0, 0, 0
            CPU_t, MEM_t = None, None

            for t in range(W):
                if t == 1 and now+W-1 >= lenx:
                    break
                candidate = {}
                
                if t != 0 and flag:
                    findOV = self.findOverAndUnder(cluster, b, y, t)
                flag = False
                
                for _ in range(K):
                    CPU_t, MEM_t, flag = self.RandomGreedySimplify_new(
                        M, a, b, v, t, findOV, candidate, CPU_t, MEM_t)
                    
                    if flag:

                        break
                    candidate.clear()
                
                if t == 0:
                    candidate_copy[z] = candidate
                if len(candidate) == 0 or flag == False:
                    cost += cluster.NoMigration(t)
                    
                    if t == 0:
                        mig, bal, value = 0, cost, cost
                    continue
                
                migx, balx, valuex = cluster.costForMigration(
                    candidate, now, t, W, b, a, M)
                cost += valuex
                
                if t == 0:
                    mig, bal, value = migx, balx, valuex
                
                if cost > cost_min:
                    break
            
            if cost < cost_min:
                cost_min = cost
                min_z, balf, migf, valuef = z, bal, mig, value
            cluster.backZero(z, now, W)

        motivation = cluster.freshStructPmVm(candidate_copy, min_z, now, W, b)
        
        if len(motivation) > 0:
            motivationFile = self.motivationFile
            
            with open(motivationFile, "a") as f:
                writer = csv.writer(f)
                
                for containerId, metric in motivation.items():
                    writer.writerow([now, containerId, metric])

        return min_z, balf, migf, valuef

    
    def writeCompare(self, name, data, files):
        filename = "/hdd/lsh/Scheduler/runthis/compare/"+files
        file = open(filename, 'a')
        file.write(name)
        file.write(str(data))
        file.write("\n")
        file.close()

    
    def findOverAndUnder(self, cluster: Cluster, b, y, t):
        pm_cpu_t = {k: cpusumlist[t]
                    for k, cpusumlist in cluster.cpusum.items()}
        pm_mem_t = {k: cpusumlist[t]
                    for k, cpusumlist in cluster.memsum.items()}

        params = self.params
        allcpuValue = np.array(list(pm_cpu_t.values()))
        allmemValue = np.array(list(pm_mem_t.values()))
        avg_CPU = np.sum(allcpuValue) / params["M"]
        avg_MEM = np.sum(allmemValue) / params["M"]
        max_CPU = max(allcpuValue)
        max_MEM = max(allmemValue)

        thr_CPU = y * (max_CPU - avg_CPU) + avg_CPU
        thr_MEM = y * (max_MEM - avg_MEM) + avg_MEM

        cpu_t = cluster.vm_cpu[:, t]
        mem_t = cluster.vm_mem[:, t]
        cpumem = np.vstack((cpu_t, mem_t)).T
        cpumem_desc = cpumem[np.lexsort(cpumem[:, ::-1].T)]

        thresh_out = (thr_CPU ** 2 + b * thr_MEM ** 2) / 2
        thresh_in = (avg_CPU ** 2 + b * avg_MEM ** 2) / 2

        cpu_sum = 0
        mem_sum = 0
        
        for i in cpumem_desc:
            cpu_sum = cpu_sum + i[0]
            mem_sum = mem_sum + i[1]
            
            if cpu_sum < avg_CPU and mem_sum < avg_MEM:
                temp = (i[0] ** 2 + b * i[1] ** 2) / 2
                thresh_in = thresh_in - temp
            else:
                temp = (
                    (avg_CPU - cpu_sum + i[0]) ** 2 + b * (avg_MEM - mem_sum + i[1]) ** 2) / 2
                thresh_in = thresh_in - temp
                break

        allVmCsPluMs = cluster.pm_cost

        bal = {pmid: v[t] for pmid, v in allVmCsPluMs.items()}

        pmids = np.array(list(bal.keys()))
        v = np.array(list(bal.values()))
        overv = np.where(v > thresh_out)[0]
        over = pmids[overv]
        underv = np.where(v < thresh_in)[0]
        under = pmids[underv]
        
        return over, under

    
    def RandomGreedySimplify_new(self, M, a, b, v, t, findOV, candidate, CPU_t=None, MEM_t=None):
        cluster = self.cluster
        machines = cluster.machines
        cpusum = cluster.cpusum
        memsum = cluster.memsum

        cpu_t = list(cluster.vm_cpu[:, t])
        mem_t = list(cluster.vm_mem[:, t])

        over, under = findOV
        
        if CPU_t is None or MEM_t is None:
            CPU_t = {k: cpusumlist[t] for k, cpusumlist in sorted(
                cpusum.items(), key=lambda x: x[0])}
            MEM_t = {k: memsumlist[t] for k, memsumlist in sorted(
                memsum.items(), key=lambda x: x[0])}
            CPU_t = np.array(list(CPU_t.values()))
            MEM_t = np.array(list(MEM_t.values()))

        for s in over:
            machinethis = machines[s]
            instances = machinethis.instances
            mig_candi_s = np.array([x for x in instances.keys()])

            samples = np.ceil(v*len(mig_candi_s))
            samples = int(samples)
            lhd = lhs(1, samples)
            mig_loc = lhd * len(mig_candi_s)
            mig_loc = mig_loc[:, 0].astype(int)

            mig = np.unique(mig_candi_s[mig_loc]).astype(int)
            
            for m in mig:
                destination = s
                m = int(m)

                bal_d_cpu = cpu_t[m] * (CPU_t[s] - cpu_t[m] - CPU_t[under])
                bal_d_mem = mem_t[m] * (MEM_t[s] - mem_t[m] - MEM_t[under])

                bal_d = np.array(bal_d_cpu + b * bal_d_mem)
                mig_m = np.array(a * (M-1) * mem_t[m])
                idx = np.array(np.where(bal_d > mig_m)[0])
                lendx = len(idx)
                
                if lendx == 0:
                    continue
                allmetric = bal_d

                tmps = {under[idx[i]]: allmetric[idx[i]] for i in range(lendx)}
                candiUnder = [k for k, v in sorted(
                    tmps.items(), key=lambda x:x[1], reverse=True)]

                for destination in candiUnder:
                    rescpu = CPU_t[destination]+cpu_t[m]
                    resmem = MEM_t[destination]+mem_t[m]
                    
                    if destination != s and \
                            rescpu < machinethis.cpu_capacity and \
                            resmem < machinethis.mem_capacity:
                        CPU_t[s] -= cpu_t[m]
                        CPU_t[destination] = rescpu
                        MEM_t[s] -= mem_t[m]
                        MEM_t[destination] = resmem
                        
                        if m not in candidate:
                            candidate[m] = [(s, destination)]
                        else:
                            candidate[m].append((s, destination))
                        break

        if len(candidate) > 0:
            for k in machines.keys():
                if machines[k].cpu_capacity < CPU_t[k] or machines[k].mem_capacity < MEM_t[k]:
                    return None, None, False
        
        if len(candidate) <= 0:
            return None, None, False
        
        return CPU_t, MEM_t, True
