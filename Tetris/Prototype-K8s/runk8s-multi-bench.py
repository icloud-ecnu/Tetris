from concurrent.futures import process
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from io import StringIO, BytesIO
from real.sxyAlgo.algorithm import Scheduler
from time import sleep,time
from sys import exit
import csv
from operator import eq
import re
from multiprocessing import Process,Pool, cpu_count,Manager

class ScheduleSys:
    def __init__(self,algo) -> None:
        self.cpudict = {} # podname:cpulist
        self.memdict = {} # podname:memlist
        self.algorithm = Scheduler()
        
        self.algoName = algo
        self.nodes = {} # all of nodename:set(podname)
        self.pods = set() # all of podname
        self.startTime = time()
        # self.getPodNodNum()
        self.cpumax = 5
        self.memmax = 10
        pass
    
    def schedule(self):
        t = 0
        end = 3 if self.algorithm == "tetris" else 5
        # os.system("bash ./request.sh")
        flag =True # self.checkNodeAndPodCmd()
        test = True
        
        while flag:
            # self.checkNodeAndPodCmd()
            print(f"################################# {t}th loading #############################")
            # if t!=0:
            #     for i in range(podnum):
            #         for podname in self.cpudict.keys():
            #             self.getCpuMemNow(podname,t)
            
            # if self.algoName == "tetris" and test :
            #     for i in range(9):
            #         print("read time at",i)
            #         self.getPodNodNum(t,False)
            #         t = t+1
            #     test = False
            
            self.getPodNodNum(t)
            sleep(8)
            print("sub process done")
            # self.NodeToPod(t)
            # print(f"pods is {self.pods}\n nodes= \n{self.nodes}")
            assert len(self.pods) > 0 and len(self.nodes) ==10
            if t==0:
                Filename = './metric/sandpiper.csv' if algo == "sandpiper" else './metric/tetris.csv'
                with open( Filename,'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(["value","eval_bal","eval_mig"])
            self.testprint(self.nodes,"old nodes")

            if self.algoName == "sandpiper":
                newnodes ,eval_mig= self.algorithm(self.nodes,self.cpudict,self.memdict,self.algoName)
            elif self.algoName == "tetris":
                newnodes,eval_mig = self.algorithm(self.nodes,self.cpudict,self.memdict,self.algoName,self.cluster,t+10)
            elif self.algoName == "drl":
                pass
            else:
                print("choose the scheduling algorithm --algo=sandpiper or --algo=tetris\n done")
                exit()
            
            # whether need migrate
            self.testprint(newnodes,"after algo newnodes")
            
            if  eval_mig == 0 or self.nodes == newnodes or eq(self.nodes,newnodes):
                
                sleep(5)
                # flag = self.checkNodeAndPodCmd()
                t=t+1
                if t>end:
                    print(f"at {t} Done")
                    break

                continue
            # print(f"after schedule nodes= \n {self.nodes}")
            self.old_pod_name = {}
            for k,v in self.nodes.items():
                for e in v:
                    self.old_pod_name[e] = k

            assert len(self.nodes)==self.nodeNum and len(self.pods) == self.podNum
            self.modifyYaml(newnodes) # migrate
            
            t=t+1

            if t>end:
                print(f"at {t} Done")
                break
    
    
    def getCpuMemNow(self,podname,podeCpuMem,t=0):
        """obtain the resource consumption"""
        # kubectl top pod | grep tc0 | awk '{print $3}' | tr -cd '[0-9]'
        cpu,mem = '',''
        looptimes = 0
        
        while cpu=='' or mem=='':
            # print(f"podname={podname} In getCpuMemNow spending cpu={cpu},mem={mem}")
            sleep(3)
            looptimes += 1
            if looptimes > 5:
                with os.popen("kubectl top pod | grep '"+podname+" '") as p:
                    print("wrong",p.read())
                exit(1)
            with os.popen("kubectl top pod | grep '"+podname+" ' | awk '{print $2}' | tr -cd '[0-9]'") as cmdcpu :
                cpu = cmdcpu.read()
            with os.popen("kubectl top pod | grep '"+podname+" ' | awk '{print $3}' | tr -cd '[0-9]'") as cmdmem :
                mem = cmdmem.read()
            
        # print(f"podname={podname} cpu: {cpu} mem:{mem}")
        intcpu = int(cpu)
        intmem = int(mem)
        # calculate the percentage of resource consumption
        cpuperc = intcpu / 20 # =intcpu/2000*100
        memperc = intmem / 4096 * 100 # =intmem/(4*1024)*100
        podeCpuMem[podname] = (cpuperc , memperc )
        return 
        # print("intcpu --- ",intcpu)
        # assert 1==0
        # get node:containers
       
    
    
    # delete the pod - modify pod.yaml - recreate the pod
    def modifyYaml(self,nodes):
        # migrated_pod_list = self.pods # need to be migrated
       
        print("migration start")
        # nodes = self.nodes
        
        p =1
        
        if self.algoName == "tetris":
            print(self.podnameToidx,self.nodenameToidx)
            # assert 1==0
        
        for node_name,migrated_pod_list in nodes.items():
            if self.algoName == "tetris":
                cnode = self.clusternodes[self.nodenameToidx[node_name]]
            
            for pod_name in migrated_pod_list:
                if self.algoName == "tetris":
                    container = self.clusetrpods[self.podnameToidx[pod_name]]
                    lastnode = self.clusternodes[container.mac_id]
                    lastnode.pop(container.id)
                    cnode.push(container)
                if pod_name[0:2] != 'tc' and self.old_pod_name[pod_name]==node_name:
                    print(f"ready to migrate {pod_name} to {node_name}")
                if pod_name[0:2] != 'tc' or self.old_pod_name[pod_name]==node_name:
                    continue
                if p==1:
                    p =0
                    
                    print("modify ",pod_name )
                    # assert 1==0
                    # po = os.popen("kubectl delete pod "+pod_name+"& ""sed -i '4c\  name: "+pod_name+"' /root/tomcat/pod.yaml"\
                    #     "& sed -i '8c\  nodeName: "+node_name+"' /root/tomcat/pod.yaml"\
                    #         "& kubectl create -f /root/tomcat/pod.yaml") # delete the pod_name
                    with os.popen("kubectl delete pod "+pod_name) as po:
                        print(f"delete {pod_name} from {self.old_pod_name[pod_name]} ",po.read())
                        
                        
                    with os.popen("sed -i '4c\  name: "+pod_name+"' /root/tomcat/pod.yaml") as po:
                        print(f"modify /root/tomcat/pod.yaml for {pod_name}",po.read())
                        
                        sleep(0.1)
                    
                    with os.popen("sed -i '8c\  nodeName: "+node_name+"' /root/tomcat/pod.yaml") as po:
                        print(f"modify /root/tomcat/pod.yaml for {node_name}",po.read())
                       
                        sleep(0.1)
                    
                    with os.popen("kubectl create -f /root/tomcat/pod.yaml") as po:
                        print(f"create /root/tomcat/pod.yaml for {node_name}_{pod_name}",po.read())
                        
                        sleep(0.1)
                    
                if po !=None:
                    p = 1
        
                # os.popen("sed -i '4c\  name: "+pod_name+"' /root/tomcat/pod.yaml") # 改为pod_name
                # os.popen("sed -i '8c\  nodeName: "+node_name+"' /root/tomcat/pod.yaml") # 改为node_name
                # os.popen("kubectl create -f /root/tomcat/pod.yaml")
        #sleep(5)
    
    
    def checkNodeAndPodCmd(self):

        cmd = os.popen("kubectl get pod ")
        
        print(f"\nat time {time()-self.startTime}: \n{cmd.read()} \n")
        print(self.cpudict)
        return True
    
    
    def getPodNodNum(self,t,flag=True):

        with os.popen("kubectl get pod -o wide") as po:
            res = po.read()
            lines = res.split("\n")
        n = len(lines)-1
        ps =  []
        podeCpuMem = Manager().dict()
        cpudict  = self.cpudict
        memdict = self.memdict
        
        for i in range(1,n):
            idx,_= re.search(" ",lines[i]).span()
            podname = lines[i][0:idx]

            if podname not in cpudict:
                cpudict[podname] =[]
            if podname not in memdict:
                memdict[podname] = []
            ps.append(Process(target=self.getCpuMemNow,args=(podname,podeCpuMem)))
        
        for p in ps:
            p.start()
        
        for p in ps:
            p.join()
        
        if flag:
            print("finish in pod and node")
        # done
        
        if t == 0 and self.algoName == "tetris":
            from sxyAlgo.cluster import Cluster
            self.cluster = Cluster()
            cluster = self.cluster
            from sxyAlgo.container import Container
            from sxyAlgo.node import Node
        # print("podecpimem len",len(podeCpuMem))
        # assert 1==0
        
        for podname,v in podeCpuMem.items():
            # print(podname,v)
            cpuperc,memperc = v
            
            if t == 0 and self.algoName == "tetris":
                self.cpudict[podname] = [1.65, 1.3, 1.45, 0.95, 1.5, 1.8, 1.6, 1.55, 1.25, 1.65]
                self.memdict[podname] = [2.24609375, 2.24609375, 2.24609375, 2.24609375, 2.24609375, 2.24609375,\
                                         2.24609375, 2.24609375, 2.24609375, 2.24609375]
            self.cpudict[podname].append(cpuperc)
            self.memdict[podname].append(memperc)
        
        # print(self.cpudict)
        # assert 1==0
        self.podNum = len(self.cpudict)
        
        with os.popen("kubectl get node -o wide") as po:
            res = po.read()
            lines = res.split("\n")
        
        # return True
        pods = self.pods=set()
        nodes = self.nodes={}
        # nodeid = 1
        
        if self.algoName == "tetris":
            cluster = self.cluster
        if flag:
            print("container len",len(self.cpudict),self.cpudict.keys())
        # assert 1==0
        pod_id = 0
        
        for podname in self.cpudict.keys():
            with os.popen("kubectl get pod -o wide|grep '"+podname+ " ' | awk '{print $9}'") as p :
                pods.add(podname)
                nodename = p.read()[0:-1] # k8s-node1 k8s-node2 etc.
            if nodename == "<none>":
                with os.popen("kubectl get pod -o wide|grep '"+podname+ " ' | awk '{print $7}'") as p :
                    nodename = p.read()[0:-1]
            if nodename not in self.nodes:
                nodes[nodename]=set()
            
            # pod_id = 0 #int(podname[-1:])
            cpu = self.cpudict[podname][-1]
            mem = self.memdict[podname][-1]
            
            if  self.algoName == "tetris" :
                node_id = int(nodename[8:])-1
                if  t==0:
                    if node_id not in cluster.nodes:
                        node_config = {"nodeName":nodename,"id":node_id ,"cpu_capacity":self.cpumax,"mem_capacity":self.memmax}
                        node = Node(node_config)
                        cluster.nodes[node.id] = node
                        node.attach(cluster)
                    else:
                        node = cluster.nodes[node_id]
                    container_config = {"containerName":podname,"id":pod_id,"node_id":node_id,\
                        "cpu":cpu,"mem":mem,\
                            "memory_curve":[v for v in self.memdict[podname]],"cpu_curve":[v for v in self.cpudict[podname]]}
                    container = Container(container_config)
                    cluster.containers[container.id] = container
                    
                    assert container.mac_id == node_id
                    # node = cluster.nodes.get(nod_id, None)
                    node = cluster.nodes[node_id]
                    assert node is not None
                    node.push(container)
                    
                    container.memlist.append(mem)
                    container.cpulist.append(cpu)
                
                if  t!=0:
                    # the new node
                    if node_id not in cluster.nodes:
                        node_config = {"nodeName":nodename,"id":node_id ,"cpu_capacity":self.cpumax,"mem_capacity":self.memmax}
                        node = Node(node_config)
                        cluster.nodes[node.id] = node
                        node.attach(cluster)
                    else:
                        node = cluster.nodes[node_id]
                    
                    # the new pod
                    if podname not in self.podnameToidx:
                        pod_id = max(list(cluster.containers.keys()))+1
                        container_config = {"containerName":podname,"id":pod_id,"node_id":node_id,\
                        "cpu":cpu,"mem":mem,\
                            "memory_curve":[v for v in self.memdict[podname]],"cpu_curve":[v for v in self.cpudict[podname]]}
                        container = Container(container_config)
                        cluster.containers[container.id] = container
                        node.push(container)
                    else:
                        container =cluster.containers[self.podnameToidx[podname]] 
                        # the container has been migrated
                        if container.id not in node.containers:
                            cluster.nodes[container.node.id].pop(container.id)
                            node.push(container)
                    container.memlist.append(mem)
                    container.cpulist.append(cpu)
                
                pod_id+=1 # sxyalgo
            
            # if t==0:
            # nodes and podes one-to-one correspondence
            nodes[nodename].add(podname)
            

        # test tetris algo
        if self.algoName == "tetris":
            print("test nodes and containers")
            for k,node in cluster.nodes.items():
                containers = node.containers
                print("node.nodename,node.id,k ",node.nodename,node.id,k)
                
                # print([(c.id,c.name) for c in containers.values()])

        self.nodeNum = len(nodes)
            # print("cluster container = {0}".format(len(cluster.containers.keys())))
        
        if self.algoName == "tetris":
            print("len of containers = {0}".format(len(cluster.containers)))
            print("len of nodes = {0}".format(len(cluster.nodes)))
            print("container = {0} node = {1}".format(cluster.containers.keys(),cluster.nodes.keys()))
        # print({ k:v.id for k, v in cluster.nodes.items()},{k:v.name for k,v in cluster.containers.items()})
        # assert 1==0
            
            self.podnameToidx = {}
            self.nodenameToidx= {}
            
            clusetrpods = self.cluster.containers
            clusternodes = self.cluster.nodes

            for podid ,pod in clusetrpods.items():
                self.podnameToidx[pod.name]=podid

            for nodid,nod in clusternodes.items():
                self.nodenameToidx[nod.nodename] = nodid
        i=0
        self.podnameToidx = {}
        self.idxTopodname = {}
        
        for nodename,pods in nodes.items():
            for podname in pods:
                self.podnameToidx[podname]=i
                self.idxTopodname[i] = podname
                i = i+1
        self.algorithm.addChange(self.podnameToidx,self.idxTopodname )
        self.test(t)
    
    
    def test(self,t):
        cpu = self.cpudict
        mem = self.memdict
        print("test---- nodes len  =  ",len(self.nodes))
        
        for podnamecpu,cpuv, in cpu.items():
            print("{0}:[ cpuv = {1} memv = {2} ]".format(podnamecpu,cpuv,mem[podnamecpu]))
        pass
    
    
    def testprint(self,iteras,strs):
        print(strs)
        
        for k,v in iteras.items():
            print(k,v)
        pass

    
if __name__ =="__main__":
    import argparse
    
    parse = argparse.ArgumentParser()
    parse.add_argument("--algo",type=str)
    args = parse.parse_args()
    algo = args.algo
    test = ScheduleSys(algo)
    # sleep(10)
    test.schedule()
    
