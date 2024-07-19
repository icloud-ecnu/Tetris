import time
from typing import Dict
import numpy as np
from machine import Machine
from instance import Instance

class Cluster(object):
    def __init__(self):
        self.machines = {}
        self.instances = {}  
        
        self.cpusum = None  
        self.memsum = None
        self.cpusum_copy = None
        self.memsum_copy = None
        
        self.vm_cpu = None
        self.vm_mem=None
        
        self.pm_cost= None
        self.pm_cost_copy = None
        self.t_pm_cost_motivatin = {}

        self.modifyPmCopy=None
        self.driftPm = {}
        
        
        self.over = None
        self.under = None
        self.candidate_machine = None
        self.candidate_container = None
        self.candidate_finanlly = []
        self.oldmac_containers = {}
        
    
    def attachOverUnder(self,over,under):
        x = len(over) 
        y =  len(under)
        
        self.all_candidate_c = {}
        self.over= over[:x]
        self.candidate_machine = { k: self.machines[k] for k in self.over }
        self.under = under[:y]
        self.candidate_container = {  k:self.instances[k] for k in self.under}
        
    
    def add_old_new(self, mac_ids, inc_ids):
        self.mac_ids = mac_ids
        self.inc_ids = inc_ids
  
    
    def copy_machine(self,machines):
        self.machines= machines.copy()
    
    
    def copy_instances(self,instances):
        self.instances = instances.copy()
    
    
    def configure_machines(self, machine_configs: dict):
        for machine_config in machine_configs.values():
            machine = Machine(machine_config)
            self.machines[machine.id] = machine
            machine.attach(self)

    
    def configure_instances(self, instance_configs: dict):
        for instance_config in instance_configs.values():
            inc = Instance(instance_config)
            self.instances[inc.id] = inc
            
            machine_id = inc.mac_id
            machine = self.machines.get(machine_id, None)
           
            assert machine is not None
            machine.push(inc)
    
    
    def cost_all_pm(self,clock,w,b):
        machines = self.machines
        cost_min = 0
        
        for pm in machines.values(): 
            v = pm.cost_first(clock,w,b)
           
            cost_min += v
        return cost_min
    
    
    def cost_all_pm_first(self,clock,w,b):
        cost_min = 0
        cpusum = self.cpusum = {}
        memsum = self.memsum = {}
        vm_cpu = {}
        vm_mem = {}
        pm_cost = {}
        machines = self.machines
        bal = 0
        
        for pm in machines.values(): 
            v = pm.getnowPluPredictCost(clock,w,b)
            pm_cost[pm.id] = pm.CsPluMs 
            
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
        
        print(len(vm_cpu),len(self.instances),len(vm_cpu))
        assert len(self.vm_cpu) == len(self.instances)
        self.pm_cost = pm_cost
        self.pm_cost_copy = {k:v for k,v in pm_cost.items() }
        self.cpusum_copy = {k:v for k,v in cpusum .items()}
        self.memsum_copy = {k:v for k,v in memsum .items()}
        self.modifyPmCopy = []
        self.t_pm_cost_motivatin[clock] = {k:v for k,v in pm_cost.items() }
        
        return cost_min,bal
    
    
    def costForMigration(self,candidate:Dict,clock,t,w,b,a,M):
        machines = self.machines
        instances = self.instances
        mig = 0
        bal = 0
        mac_modify = set()
        cpusum = self.cpusum
        memsum = self.memsum
        pm_cost = self.pm_cost
          
        candidate_s_d = [ x for v in list(candidate.values()) for x in v[-1]]
        mac_modify.update(candidate_s_d)
        otherPmCost = [ v[t] for k,v in pm_cost.items() if k not in mac_modify ]
        
        assert len(otherPmCost) + len(mac_modify) == len(machines)
        otherPmCostSum = np.sum( otherPmCost )
        
        for vmid,slou in candidate.items():
            s,destination=slou[-1]
            mig += machines[s].migrateOut(vmid,t)
            machines[destination].migrateIn(instances[vmid],t)
        
        for macid in mac_modify:
            bal+=machines[macid].afterMigration_cost(clock,t,w,b)
            cpusum[macid] = machines[macid].cpu_sum_w
            memsum[macid] = machines[macid].mem_sum_w
            pm_cost[macid] = machines[macid].CsPluMs_migraton 
        
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
        
        machines = self.machines
        instances = self.instances
        self.pm_cost = {k:v for k,v in pm_cost_copy.items() }
        self.cpusum = {k:v for k,v in cpusum_copy.items()}
        self.memsum = {k:v for k,v in memsum_copy.items()}
        
        print(f"len of modifyPmCopy: {len(self.modifyPmCopy)}")
        macids = set()
        
        if len(self.modifyPmCopy) > 0:
            x = range(len(self.modifyPmCopy)-1,-1,-1)
            
            for i in x:
                candidate = self.modifyPmCopy[i]
                
                for vmid,v in candidate.items():
                    destination,s = v[0]
                    machines[s].pop(vmid)
                    machines[destination].push(instances[vmid])
                    macids.add(s)
                    macids.add(destination)
        
        for macid in macids:
            machines[macid].getEveryTimeCpuList(clock,w)
        
        self.modifyPmCopy = []
        
    
    # def freshStructPmVm(self,candidate_copy,z,clock,w,b):
    #     if z==-1:
    #         return {}
        
    #     candidate = candidate_copy[z]
        
    #     if len(candidate) == 0:
    #         print("no migration")
    #         return {}
        
    #     machines = self.machines
    #     instances = self.instances
    #     outpm = {v[0][0]:0 for k,v in candidate.items() }
    #     outpm = {macid:set([v for v in machines[macid].instances.keys()])for macid in outpm.keys()}
    #     inpm = {v[0][1]:0 for k,v in candidate.items() }
    #     inpm = {macid:set([v for v in machines[macid].instances.keys()])for macid in inpm.keys()}
    #     print(f'outpm is {outpm}')
    #     print(f'inpm is {inpm}')
    #     motivation = {}
    #     moti_len = 2
        
    #     for vmid,v in candidate.items():
    #         if (len(v)>1):
    #             print(v)
    #             assert 1==0
            
    #         s,destination = v[0]
    #         before_value = []
    #         after_value = []
            
    #         # before
    #         try:
    #             for t in range(moti_len):
    #                 beforePmOutCost = machines[s].afterOneContainerMigration(clock+t,w,b)
    #                 beforePmInCost = machines[destination].afterOneContainerMigration(clock+t,w,b)
    #                 before_value.append(beforePmOutCost+beforePmInCost)
    #                 print(f"计算 before_value 耗时 {time() - start_time:.2f} 秒")
    #         except Exception as e:
    #             print(f"计算 before_value 出错: {e}")
    #         machines[s].pop(vmid)
    #         machines[destination].push(instances[vmid])
            
    #         try:
    #             for t in range(moti_len):
    #                 afterPmOutCost = machines[s].afterOneContainerMigration(clock+t,w,b)
    #                 afterPmInCost = machines[destination].afterOneContainerMigration(clock+t,w,b)
    #                 after_value.append(afterPmOutCost+afterPmInCost)
    #         except:
    #             print()
            
    #         motivation[vmid] = [s,destination,before_value,after_value]  
        
    #     afteroutpm = {macid:set([v for v in machines[macid].instances.keys()])for macid in outpm.keys()}
    #     afterinpm = {macid:set([v for v in machines[macid].instances.keys()])for macid in inpm.keys()}
        
    #     diffout = {macid:v.difference(afteroutpm[macid])for macid,v in outpm.items()}
    #     diffin = {macid:afterinpm[macid].difference(v)for macid,v in inpm.items()}
        
    #     violations = self.isAllUnderLoad(clock)
        
    #     self.driftPm[clock]={"outpm":outpm,"afteroutpm":afteroutpm,"inpm":inpm,"afterinpm":afterinpm,"diffout":diffout,"diffin":diffin,"violations":violations}
    #     return motivation
    
    def freshStructPmVm(self, candidate_copy, z, clock, w, b):
        if z == -1:
            return {}

        candidate = candidate_copy[z]
        if len(candidate) == 0:
            print("no migration")
            return {}

        from time import time

        machines = self.machines
        instances = self.instances

        outpm = {v[0][0]: 0 for k, v in candidate.items()}
        outpm = {macid: set([v for v in machines[macid].instances.keys()]) for macid in outpm.keys()}
        inpm = {v[0][1]: 0 for k, v in candidate.items()}
        inpm = {macid: set([v for v in machines[macid].instances.keys()]) for macid in inpm.keys()}
        print(f'outpm is {outpm}')
        print(f'inpm is {inpm}')

        motivation = {}
        moti_len = 2

        for vmid, v in candidate.items():
            if len(v) > 1:
                print(v)
                assert 1 == 0

            s, destination = v[0]
            before_value = []
            after_value = []

            # before
            try:
                start_time = time()
                for t in range(moti_len):
                    beforePmOutCost = machines[s].afterOneContainerMigration(clock + t, w, b)
                    beforePmInCost = machines[destination].afterOneContainerMigration(clock + t, w, b)
                    before_value.append(beforePmOutCost + beforePmInCost)
                print(f"计算 before_value 耗时 {time() - start_time:.2f} 秒")
            except Exception as e:
                print(f"计算 before_value 出错: {e}")

            start_time = time()
            machines[s].pop(vmid)
            machines[destination].push(instances[vmid])
            print(f"迁移容器 {vmid} 耗时 {time() - start_time:.2f} 秒")

            try:
                start_time = time()
                for t in range(moti_len):
                    afterPmOutCost = machines[s].afterOneContainerMigration(clock + t, w, b)
                    afterPmInCost = machines[destination].afterOneContainerMigration(clock + t, w, b)
                    after_value.append(afterPmOutCost + afterPmInCost)
                print(f"计算 after_value 耗时 {time() - start_time:.2f} 秒")
            except Exception as e:
                print(f"计算 after_value 出错: {e}")

            motivation[vmid] = [s, destination, before_value, after_value]

        afteroutpm = {macid: set([v for v in machines[macid].instances.keys()]) for macid in outpm.keys()}
        afterinpm = {macid: set([v for v in machines[macid].instances.keys()]) for macid in inpm.keys()}

        diffout = {macid: v.difference(afteroutpm[macid]) for macid, v in outpm.items()}
        diffin = {macid: afterinpm[macid].difference(v) for macid, v in inpm.items()}

        violations = self.isAllUnderLoad(clock)

        self.driftPm[clock] = {
            "outpm": outpm, "afteroutpm": afteroutpm, "inpm": inpm,
            "afterinpm": afterinpm, "diffout": diffout, "diffin": diffin,
            "violations": violations
        }

        return motivation

    def isAllUnderLoad(self,clock,sand=False):
        machines = self.machines
        cpusum = self.cpusum
        memsum = self.memsum
        cpu_capacity = 20 if sand else 30
        
        try:
            violations = {mac.id:machines[mac.id].instances.keys() for mac in machines.values() if cpusum[mac.id][0] >cpu_capacity or memsum[mac.id][0]>mac.mem_capacity }
        except:
            violations = {}
            print("wrong")
           
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
                  
                    'cpu': m.cpu,
                    'memory': m.mem,
                   
                    'instances': {
                        j: {
                            'cpu': inst.cpu,
                            'mem': m.mem,
                        } for j, inst in m.instances.items()
                    }
                }
                for i, m in self.machines.items()
            }]
