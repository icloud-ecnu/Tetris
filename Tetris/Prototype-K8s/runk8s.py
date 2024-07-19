from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from io import StringIO, BytesIO
# from real.sxyAlgo.algorithm import Scheduler
from tetrisAlgo.algorithm import Scheduler
from time import sleep,time
from sys import exit
import csv
from operator import eq

class ScheduleSys:
    def __init__(self,algo) -> None:
        self.cpudict = {} # podname:cpulist
        self.memdict = {} # podname:memlist
        # self.nodeNum = nodeNum
        # self.podNum = podNum
        self.getPodNodNum()
        self.algorithm = Scheduler()
        self.algoName = algo
        self.nodes = {}
        self.pods = set()
        self.startTime = time()
        
        pass
    
    
    def schedule(self):
        t = 0
        podnum = self.podNum
        
        # os.system("bash ./request.sh")
        flag = self.checkNodeAndPodCmd()
        
        while flag:
            print(f"################################# {t}th loading #############################")
            
            # 获取容器在当前时间点的CPU和内存使用情况
            for i in range(podnum):
                podname = "tc"+str(i)
                self.getCpuMemNow(podname,t)
            
            # 将容器与节点进行关联
            self.NodeToPod(t)
            # print(f"pods is {self.pods}\nfirst nodes= \n{self.nodes}")
            # assert 1==0
            # run8s.log
            
            # 创建一个CSV文件，写入表头信息
            if t==0:
                Filename = './metric/sandpiper.csv' if algo == "sandpiper" else './metric/sxy.csv'
                
                with open( Filename,'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(["value","eval_bal","eval_mig"])
            
            # 根据algoName的取值选择相应的调度算法
            if self.algoName == "sandpiper":
                newnodes = self.algorithm(self.nodes,self.cpudict,self.memdict,self.algoName) #调度算法后生成新的node pod分配方案
            
            elif self.algoName == "sxy":
                # if t>20:
                #     newnodes = self.algorithm(self.nodes,self.cpudict,self.memdict,self.algoName,self.cluster,t)
                # else:
                #     newnodes = self.nodes
                newnodes = self.algorithm(self.nodes,self.cpudict,self.memdict,self.algoName,self.cluster,t)
            
            elif self.algoName == "drl":
                pass
            
            else:
                print("choose the scheduling algorithm --algo=sandpiper or --algo=sxy\n done")
                exit()
            
            # 新方案与原方案一致
            if  self.nodes == newnodes or eq(self.nodes,newnodes):
                
                sleep(5)
                flag = self.checkNodeAndPodCmd()
                t=t+1
                # if t>5:
                #     break;
                continue
            
            self.nodes = newnodes
            print(f"after schedule nodes= \n {self.nodes}")
            
            assert len(self.nodes)==self.nodeNum and len(self.pods) == self.podNum
            self.modifyYaml() # migration
            sleep(5)
            flag = self.checkNodeAndPodCmd()
            
            sleep(60)
            t=t+1

            if t>5:
                print(f"at {t} Done")
                break
    
    # 将容器与节点进行关联
    def NodeToPod(self,t):
        algoName = self.algoName
        
        if t == 0 and algoName == "sxy":
            isSxy  = True
        
        else:
            isSxy = False
        
        if isSxy:
            # TODO cluster node container
            from sxyAlgo.cluster import Cluster
            self.cluster = Cluster()
            cluster = self.cluster
            from sxyAlgo.container import Container
            from sxyAlgo.node import Node
        
        pods = self.pods
        nodes = self.nodes
        
        for podname in self.cpudict.keys():
            with os.popen("kubectl get pod -o wide|grep '"+podname+ " ' | awk '{print $9}'") as p :
                pods.add(podname)
                nodename = p.read()[0:-1] # k8s-node1 k8s-node2 etc.
            
            if nodename == "<none>":
                with os.popen("kubectl get pod -o wide|grep '"+podname+ " ' | awk '{print $7}'") as p :
                    nodename = p.read()[0:-1]
            
            if nodename not in self.nodes:
                nodes[nodename]=set()
                
                if isSxy:
                    node_id = int(nodename[-1:])-1
                    node_config = {"nodeName":nodename,"id":node_id,"cpu_capacity":100,"mem_capacity":100}
                    node = Node(node_config)
                    cluster.nodes[node.id] = node
                    node.attach(self)
            
            pod_id = int(podname[-1:])
            cpu = self.cpudict[podname][-1]
            mem = self.memdict[podname][-1]
            
            if isSxy:
                container_config = {"containerName":podname,"id":pod_id,"node_id":node_id,\
                    "cpu":cpu,"mem":mem,\
                        "memory_curve":[],"cpu_curve":[]}
                container = Container(container_config)
                cluster.containers[container.id] = container
                node_id = container.mac_id
                node = cluster.nodes.get(node_id, None)
                
                assert node is not None
                node.push(container)
            
            if algoName == "sxy":
                    container = self.cluster.containers[pod_id]
                    container.memlist.append(mem)
                    container.cpulist.append(cpu)
            # if t==0:
            nodes[nodename].add(podname)
                
    # 获取指定podname的CPU和内存使用情况，并将获取到的数据存储到字典中
    def getCpuMemNow(self,podname,t=0):
        """resource comsumption"""
        cpudict  = self.cpudict
        memdict = self.memdict
        
        if podname not in cpudict:
            cpudict[podname] = []
        
        if podname not in memdict:
            memdict[podname] = []
        # kubectl top pod | grep tc0 | awk '{print $3}' | tr -cd '[0-9]'
        cpu,mem = '',''
        looptimes = 0
        
        # 不断尝试获取podname的CPU和内存使用情况，直到成功获取到数据为止
        while cpu=='' or mem=='':
            print(f"podname={podname} In getCpuMemNow spending cpu={cpu},mem={mem}")
            sleep(3)
            looptimes += 1
            
            # 超过5次仍然无法获取到数据，则打印错误信息，并终止程序的执行
            if looptimes > 5:
                with os.popen("kubectl top pod | grep '"+podname+" '") as p:
                    print("wrong",p.read())
                exit(1)
            
            with os.popen("kubectl top pod | grep '"+podname+" ' | awk '{print $2}' | tr -cd '[0-9]'") as cmdcpu :
                
                cpu = cmdcpu.read()
            
            with os.popen("kubectl top pod | grep '"+podname+" ' | awk '{print $3}' | tr -cd '[0-9]'") as cmdmem :
                
                mem = cmdmem.read()

        # 获取到了cpu和men值
        print(f"podname={podname} cpu: {cpu} mem:{mem}")
        intcpu = int(cpu)
        intmem = int(mem)
        # try:
        #     intcpu = int(cpu) # 转为int
        # except:
        #     cmdcpu = os.popen("kubectl top pod | grep "+podname+" | awk '{print $2}' | tr -cd '[0-9]'")
        #     intcpu = int(cpu)
        # try:
        #     intmem = int(cmdmem.read())
        # except:
        #     cmdmem = os.popen("kubectl top pod | grep "+podname+" | awk '{print $3}' | tr -cd '[0-9]'")
        #     intmem = int(cmdmem.read())
            
        
        cpuperc = intcpu / 20 # /2000*100
        memperc = intmem / 4096 * 100 # /(4*1024)*100
        
        if t == 0 and self.algoName == "sxy":
            self.cpudict[podname] = [0.05, 0.1, 0.1, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05]
            self.memdict[podname] = [0.1, 0.05, 0.05, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05]
        self.cpudict[podname].append(cpuperc)
        self.memdict[podname].append(memperc)
        # print(intcpu)
        # assert 1==0
    
    
    # delete the pod - modify pod.yaml - recreate the pod
    def modifyYaml(self):
        # migrated_pod_list = self.pods
        print("migration start")
        nodes = self.nodes
        p =1
        
        # 一个node中的所有要迁移的pod
        for node_name,migrated_pod_list in nodes.items():
            for pod_name in migrated_pod_list:
                if p==1:
                    p =0
                    print("modify ",pod_name )
                    # po = os.popen("kubectl delete pod "+pod_name+"& ""sed -i '4c\  name: "+pod_name+"' /root/tomcat/pod.yaml"\
                    #     "& sed -i '8c\  nodeName: "+node_name+"' /root/tomcat/pod.yaml"\
                    #         "& kubectl create -f /root/tomcat/pod.yaml")

                    # Checkpoint the pod using CRIU，内存检查点
                    with os.popen(f"criu dump -t $(pidof {pod_name}) -D /checkpoint/{pod_name} --leave-running") as po:
                        print("Checkpointing pod:", po.read())

                    with os.popen("kubectl delete pod "+pod_name) as po:
                        print(1,po.read())
                        
                    with os.popen("sed -i '4c\  name: "+pod_name+"' /root/tomcat/pod.yaml") as po:
                        print(2,po.read())
                        #sleep(0.1)
                    
                    with os.popen("sed -i '8c\  nodeName: "+node_name+"' /root/tomcat/pod.yaml") as po:
                        print(3,po.read())
                        #sleep(0.1)
                    
                    with os.popen("kubectl create -f /root/tomcat/pod.yaml") as po:
                        print(4,po.read())
                        #sleep(0.1)
                    
                    # Restore the pod using CRIU，内存恢复
                    with os.popen(f"criu restore -D /checkpoint/{pod_name} --shell-job") as po:
                        print("Restoring pod:", po.read())
                    
                if po != None:
                    p = 1
        
                # os.popen("sed -i '4c\  name: "+pod_name+"' /root/tomcat/pod.yaml") # pod_name
                # os.popen("sed -i '8c\  nodeName: "+node_name+"' /root/tomcat/pod.yaml") # node_name
                # os.popen("kubectl create -f /root/tomcat/pod.yaml")
        #sleep(5)
    
    # 执行kubectl get pod -o wide命令来获取Pod的详细信息，并打印输出结果和self.cpudict的值。然后，它返回布尔值True，表示函数执行成功
    def checkNodeAndPodCmd(self):

        cmd = os.popen("kubectl get pod -o wide") # 打开一个管道来执行命令并返回一个文件对象
        
        print(f"\nat time {time()-self.startTime}: \n{cmd.read()} \n")
        print(self.cpudict)
        cmd.close();
        return True
    
    
    def getPodNodNum(self):
        with os.popen("kubectl get pod -o wide") as po:
            res = po.read()
            lines = res.split("\n")
        self.podNum  = len(lines)-2
        
        with os.popen("kubectl get node -o wide") as po:
            res = po.read()
            lines = res.split("\n")
        self.nodeNum = len(lines)-3
        

if __name__ =="__main__":
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument("--algo",type=str)
    args = parse.parse_args()
    algo = args.algo
    test = ScheduleSys(algo)
    test.schedule()
    
