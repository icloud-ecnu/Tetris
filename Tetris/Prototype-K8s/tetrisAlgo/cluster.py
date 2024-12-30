
import time
from typing import Dict
import numpy as np

from tetrisAlgo.container import Container



class Cluster(object):
    def __init__(self):
        self.nodes = {}
        self.containers = {} 
        
        self.cpusum = None  
        self.memsum = None
        self.cpusum_copy = None
        self.memsum_copy = None
        
        self.vm_cpu = None
        self.vm_mem=None
        
        self.pm_cost= None
        self.pm_cost_copy = None

        self.modifyPmCopy=None
        self.driftPm = {}
    def add_old_new(self, mac_ids, inc_ids):
        self.mac_ids = mac_ids
        self.inc_ids = inc_ids
    
    def configure_nodes(self, node_configs: dict):

        for node_config in node_configs.values():
            node = node(node_config)
            self.nodes[node.id] = node
            node.attach(self)

    def configure_containers(self, container_configs: dict):
        
        for container_config in container_configs.values():
            inc = Container(container_config)
            self.containers[inc.id] = inc
            
            node_id = inc.mac_id
            node = self.nodes.get(node_id, None)
            
            assert node is not None
            node.push(inc)
   
   
    def cost_all_pm_first(self,clock,w,b):
        cost_min = 0
        cpusum = self.cpusum = {}
        memsum = self.memsum = {}
        vm_cpu = {}
        vm_mem = {}
       
        pm_cost = {}
        nodes = self.nodes
        bal = 0
        for pm in nodes.values(): 
            v = pm.getnowPluPredictCost(clock,w,b)
        
            pm_cost[pm.id] = pm.CsPluMs #
            try:
                bal+=pm_cost[pm.id][0]
            except:
                bal +=0
                pm_cost[pm.id]=np.array([0 for x in range(w)])
            
            cpusum [pm.id] = pm.cpu_sum_w
            memsum [pm.id] = pm.mem_sum_w
           
            
            vm_cpu.update(pm.cpuPluPredict)
            vm_mem.update(pm.memPluPredict)
            
            cost_min += v
        vm_cpu = sorted(vm_cpu.items(),key=lambda x:x[0])
        vm_mem = sorted(vm_mem.items(),key=lambda x:x[0])
    
        self.vm_cpu = np.array([v for k,v in vm_cpu])
        self.vm_mem = np.array([v for k,v in vm_mem])
        
        self.pm_cost = pm_cost
        
        self.pm_cost_copy = {k:v for k,v in pm_cost.items() }
        self.cpusum_copy = {k:v for k,v in cpusum .items()}
        self.memsum_copy = {k:v for k,v in memsum .items()}
        self.modifyPmCopy = []
        
        return cost_min,bal
    
    def costForMigration(self,candidate:Dict,clock,t,w,b,a,M):
        nodes = self.nodes
        containers = self.containers
        mig = 0
        bal = 0
        mac_modify = set()
        cpusum = self.cpusum
        memsum = self.memsum
        
        pm_cost = self.pm_cost
        
        candidate_s_d = [ x for v in list(candidate.values()) for x in v[-1]]
        mac_modify.update(candidate_s_d)
        otherPmCost = [ v[t] for k,v in pm_cost.items() if k not in mac_modify ]
        assert len(otherPmCost) + len(mac_modify) == len(nodes)
        otherPmCostSum = np.sum( otherPmCost )
        for vmid,slou in candidate.items():
            s,destination=slou[-1]
            mig += nodes[s].migrateOut(vmid,t)
            nodes[destination].migrateIn(containers[vmid],t)
        for macid in mac_modify:
            
            bal+=nodes[macid].afterMigration_cost(clock,t,w,b)
            cpusum[macid] = nodes[macid].cpu_sum_w
            memsum[macid] = nodes[macid].mem_sum_w
            
            pm_cost[macid] = nodes[macid].CsPluMs_migraton 
        self.modifyPmCopy.append(candidate)
        return mig,bal+otherPmCostSum,bal+(M-1)*mig*a+otherPmCostSum
    
    def NoMigration(self,t):
        pm_cost = self.pm_cost
        PmCost = np.sum([ v[t] for k,v in pm_cost.items() ])
        return PmCost
    
    
    def backZero(self,z,clock,w):
        cpusum_copy = self.cpusum_copy 
        memsum_copy = self.memsum_copy
        pm_cost_copy = self.pm_cost_copy
        self.pm_cost_copy = {k:v for k,v in pm_cost_copy.items() }
        self.cpusum_copy = {k:v for k,v in cpusum_copy.items()}
        self.memsum_copy = {k:v for k,v in memsum_copy.items()}
        
        nodes = self.nodes
        containers = self.containers
        self.pm_cost = {k:v for k,v in pm_cost_copy.items() }
        self.cpusum = {k:v for k,v in cpusum_copy.items()}
        self.memsum = {k:v for k,v in memsum_copy.items()}
        
        macids = set()
        if len(self.modifyPmCopy) > 0:
            x = range(len(self.modifyPmCopy)-1,-1,-1)
            for i in x:
                candidate = self.modifyPmCopy[i]
                for vmid,v in candidate.items():
                    destination,s = v[0]
                    nodes[s].pop(vmid)
                    nodes[destination].push(containers[vmid])
                    macids.add(s)
                    macids.add(destination)
        
        for macid in macids:
            nodes[macid].getEveryTimeCpuList(clock,w)
       
        self.modifyPmCopy = []
        
    def freshStructPmVm(self,candidate_copy,z,clock):
        if z==-1:
            return 0
        candidate = candidate_copy[z]
        if len(candidate) == 0:
            print("no migration")
            return 0
        nodes = self.nodes
        containers = self.containers
        outpm = {v[0][0]:0 for k,v in candidate.items() }
        outpm = {macid:set([v for v in nodes[macid].containers.keys()])for macid in outpm.keys()}
        inpm = {v[0][1]:0 for k,v in candidate.items() }
        inpm = {macid:set([v for v in nodes[macid].containers.keys()])for macid in inpm.keys()}
        
        for vmid,v in candidate.items():
            if (len(v)>1):
                print(v)
                assert 1==0
            s,destination = v[0]
            nodes[s].pop(vmid)
            nodes[destination].push(containers[vmid])
        afteroutpm = {macid:set([v for v in nodes[macid].containers.keys()])for macid in outpm.keys()}
        afterinpm = {macid:set([v for v in nodes[macid].containers.keys()])for macid in inpm.keys()}
        
        diffout = {macid:v.difference(afteroutpm[macid])for macid,v in outpm.items()}
        diffin = {macid:afterinpm[macid].difference(v)for macid,v in inpm.items()}
        
        violations = self.isAllUnderLoad(clock)
       
        self.driftPm[clock]={"outpm":outpm,"afteroutpm":afteroutpm,"inpm":inpm,"afterinpm":afterinpm,"diffout":diffout,"diffin":diffin,"violations":violations}
        return len(candidate)
    
    def isAllUnderLoad(self,clock,sand=False):
        nodes = self.nodes
        cpusum = self.cpusum
        memsum = self.memsum
        cpu_capacity = 20 if sand else 30
        violations = {mac.id:nodes[mac.id].containers.keys() for mac in nodes.values() if cpusum[mac.id][0] >cpu_capacity or memsum[mac.id][0]>mac.mem_capacity }
        return violations
    def plt(self,outpm,afteroutpm,clock):
        print("#"*30)
       
        for k in outpm.keys():
            res = "\t"*5+"pm["+str(k)+"]\n"
            res += "-"*20
        pass
    
    @property
    def drift_json(self):
        return [
             {
                'time': time.asctime(time.localtime(time.time()))
            },
             {
                 k:{
                    names:
                      {str(kv):str(vv) 
                       for kv,vv in value.items()
                       }
                     for names,value in v.items()
                     }
                 for k,v in self.driftPm.items()
             }
            
        ]
        
    
    @property
    def structure(self):
        return [
            {
                'time': time.asctime(time.localtime(time.time()))
            },

            {

                i: {
                    'cpu_capacity': m.cpu_capacity,
                    'memory_capacity': m.mem_capacity,
                    # 'disk_capacity': m.disk_capacity,
                    'cpu': m.cpu,
                    'memory': m.mem,
                    # 'disk': m.disk,
                    'containers': {
                        j: {
                            'cpu': inst.cpu,
                            # 'memory': inst.memory,
                            # 'disk': inst.disk
                        } for j, inst in m.containers.items()
                    }
                }
                for i, m in self.nodes.items()
            }]
