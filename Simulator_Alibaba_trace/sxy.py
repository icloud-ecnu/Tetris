from pyDOE import lhs
import numpy as np

from time import time
from numba import jit


@jit
def CostOfMigration(x_last, x_now, mem_now):
    mig = ~(x_last == x_now).all(axis = 1) # 表示是否迁移的矩阵，即[True,False,False]表示VM_1迁移了
    # print('mig:',mig)
    cost_mig = sum(mig * mem_now) # 当前时刻每VM的单位迁移开销为其此刻mem占用
    
    return cost_mig


# 计算每台机器上资源使用量【优化版】
@jit
def ResourceUsageSimplify(cpu_t, x_t):
    x_cput = x_t.copy().T
    CPU_t = np.matmul(x_cput,cpu_t)
    # CPU_t = np.sum(x_cput, axis = 0) # 各列之和
    
    return CPU_t


# 计算负载均衡开销算法【新！计算每台机器上VM各自两两相乘之和】
@jit
def CostOfMachineLoadBalanceSimplify(cpu_t, mem_t, x_t, b):
    cost_bal = 0
    bal_array = np.zeros(x_t.shape[1])
    
    for pm in range(x_t.shape[1]): # 对每一列，即每台机器
        vm = np.where(x_t[:, pm] == 1)[0] # 在该机器上的VM索引
        cpu = cpu_t[vm] # 对应VM的cpu
        mem = mem_t[vm]
        pm_cost = 0
        
        for i in range(len(vm)-1): # 每台VM与其他VM两两相乘
            c = cpu[i] * np.sum(cpu[i+1:])
            m = mem[i] * np.sum(mem[i+1:])
            cost_bal += c + b * m # cpu乘积+b*mem乘积，所有机器之和
            pm_cost += c + b * m
        bal_array[pm] = pm_cost
    
    return bal_array


# 计算负载均衡开销算法【新！计算每台机器上VM各自两两相乘之和】
@jit
def CostOfLoadBalanceSimplify(cpu_t, mem_t, x_t, b):
    cost_bal = 0
    # bal_array = np.zeros(x_t.shape[1])
    # print(x_t)
    
    for pm in range(x_t.shape[1]): # 对每一列，即每台机器
        vm = np.where(x_t[:, pm] == 1)[0] # 在该机器上的VM索引
        cpu = cpu_t[vm] # 对应VM的cpu
        mem = mem_t[vm]
        #pm_cost = 0
        
        for i in range(len(vm)-1): # 每台VM与其他VM两两相乘
            c = cpu[i] * np.sum(cpu[i+1:])
            m = mem[i] * np.sum(mem[i+1:])
            cost_bal += c + b * m # cpu乘积+b*mem乘积，所有机器之和
            #pm_cost += c + b * m
        #bal_array[pm] = pm_cost
    
    return cost_bal


@jit
def findOverAndUnder(cpu_t,x_last,mem_t,M,b,y):
    CPU_t = ResourceUsageSimplify(cpu_t, x_last) # 当前放置下各PM的资源使用量
    MEM_t = ResourceUsageSimplify(mem_t, x_last)
    avg_CPU = np.sum(cpu_t) / M # 计算当前时刻最负载均衡情况下每台机器应承载的资源均量
    avg_MEM = np.sum(mem_t) / M
    max_CPU = np.max(CPU_t) # 当前最大值
    max_MEM = np.max(MEM_t)
    
    thr_CPU = y * (max_CPU - avg_CPU) + avg_CPU
    thr_MEM = y * (max_MEM - avg_MEM) + avg_MEM
    
    cpumem = np.vstack((cpu_t, mem_t)).T # 合并为一个二维数组
    cpumem_desc = cpumem[np.lexsort(cpumem[:,::-1].T)] # 按照cpu的大小降序排序
    
    thresh_out = (thr_CPU ** 2 + b * thr_MEM ** 2) / 2
    # thresh_out = (avg_CPU ** 2 + b * avg_MEM ** 2) / 2 # 判断迁出机器候选集的标准
    
    thresh_in = (avg_CPU ** 2 + b * avg_MEM ** 2) / 2 # 判断迁入机器候选集的标准，初始化
    flag = True
    
    if flag :
        # print(type(thresh_in))
        flag  = False
    cpu_sum = 0
    mem_sum = 0
    
    for i in cpumem_desc: # 对数组中的每一行，即每一个cpu-mem对
        cpu_sum = cpu_sum + i[0]
        mem_sum = mem_sum + i[1]
        
        if cpu_sum < avg_CPU and mem_sum < avg_MEM: # 还没达到均值
            temp = (i[0] ** 2 + b * i[1] ** 2) / 2
            thresh_in = thresh_in - temp
        else: # cpu或mem之和大于等于均值，则结束循环
            temp = ((avg_CPU - cpu_sum + i[0]) ** 2 + b * (avg_MEM - mem_sum + i[1]) ** 2) / 2
            thresh_in = thresh_in - temp
            break

    bal = CostOfMachineLoadBalanceSimplify(cpu_t, mem_t, x_last, b)
    
    over = np.where(bal > thresh_out)[0] # 迁出候选集
    under = np.where(bal < thresh_in)[0] # 迁入候选集
    
    return over,under,CPU_t,MEM_t
   
   
@jit
def RandomGreedySimplify_new(M, a, b, u, v, x_last, x_t, cpu_t, mem_t, tup, y,findOV):
    over,under,CPU_t,MEM_t= findOV
    z,t,k = tup
    print(f'\t\tat z= { z } t= { t } k= { k } over : { len(over) } under { len(under) }  ')
    
    for s in over:
        mig_candi_s = np.where(x_last[:, s] == 1)[0] # 能被迁走的VM候选集
        # mig = np.random.choice(mig_candi, np.ceil(v * len(mig_candi_s)), replace = False) # 随机乱序选择
        n=1 
        samples=np.ceil(v*len(mig_candi_s))
        samples = int(samples)
        # print(samples)
        lhd = lhs(n, samples) # 拉丁超立方抽样，输出[0,1]
        mig_loc = lhd * len(mig_candi_s)
        mig_loc = mig_loc[:,0].astype(int) # 即要被迁移的contaienr的id在候选集中的位置
        mig = np.unique(mig_candi_s[mig_loc]) # 要被迁移的contaienr的id，去掉重复值
        # print(f'\t\t\t候选集 {s}: 要被迁移的contaienr: {mig}')
        
        # 对每个迁移VM贪心选择最优迁入机器
        for m in mig:
            destination = s # 目标机器初始化为原本所在的机器
            
            for d in under: # 对每台低于均值的机器
                # 假设把m迁移到d上，带来的负载均衡开销降低值
                bal_d_cpu = cpu_t[m] * (CPU_t[s] - cpu_t[m] - CPU_t[d]) # 该VM资源量*（原机器上除该VM之外的资源总量-目标机器上原本的资源总量）
                bal_d_mem = mem_t[m] * (MEM_t[s] - mem_t[m] - MEM_t[d])
                bal_d = bal_d_cpu + b * bal_d_mem
                mig_m = a * (M-1) * mem_t[m] # 该VM的迁移开销，为此时mem
                
                max_bal = mig_m # 初始化为迁移开销，则保证每个负载均衡开销节省量都要大于迁移开销才能迁入
                
                if bal_d > max_bal: # 如果当前负载均衡节省量大于历史最大节省量，则迁入该机器
                    max_bal = bal_d
                    destination = d
            
            if destination != s: # 如果要迁
                x_t[m][s] = 0 # 把该VM从原来的机器上删除，添加到目标机器上
                x_t[m][destination] = 1
    
    return x_t


@jit
def assignto(a,b):
    idx = np.where(a!=b)
    a[idx] = b[idx]


def rangeZ(Z,W,x_last,x_t0,x_t1,x_t,placement,cpu,mem,K,M, a, b, u,v,y,CPU_MAX,MEM_MAX,cost_min):
    over,under,CPU_t,MEM_t = findOverAndUnder(cpu[:, 0],x_t0,mem[:, 0],M,b,y) 
    
    for z in Z:
        cost = 0 # 当前W内全套配置方案下的总成本
        assignto(x_last,x_t0)
        assignto(x_t1,x_t0)
        assignto(x_t,x_t0) # 初始化，因为如果是第一次，则上一时刻则为x_t0，否则则为上一轮更新时所设定的x_last
        findOV = (over,under,CPU_t,MEM_t)
        s = time()
        
        for t in range(W):
            k = 0
            cpu_t = cpu[:, t] # 提取当前时刻各VM的cpu需求量
            mem_t = mem[:, t]
            flag = False
            
            while k < K and flag==False:
                k = k+1
                tup=(z,t,k)
                assignto(x_t,x_last)
                x_t = RandomGreedySimplify_new(M, a, b, u, v, x_last, x_t,cpu_t, mem_t,tup,y,findOV)
                # print('after random',(x_t==x_last).all())
                
                if (x_t==x_last).all():
                    if k == K:
                        print('\t\tnot find in t=',t)
                        assignto(x_t,x_last)
                    continue
                
                CPU_t = ResourceUsageSimplify(cpu_t, x_t)
                MEM_t = ResourceUsageSimplify(mem_t, x_t)
                flag = ((CPU_t < CPU_MAX) & (MEM_t < MEM_MAX)).all()
                
                # 备用策略
                if k == K and flag==False:
                    print('\t\tnot find in t=',t)
                    assignto(x_t,x_last)
                
            #end k

            # 计算此次迁移开销
            cost_mig = CostOfMigration(x_last, x_t, mem_t)
            # 计算此次负载均衡开销
            cost_bal = CostOfLoadBalanceSimplify(cpu_t, mem_t, x_t, b)
            # 计算当前时刻放置的总开销
            cost_t = cost_bal + a * (M-1) * cost_mig
            # 计算当前放置方案下W窗口内总开销
            cost = cost + cost_t
            
            if cost > cost_min:
                print(f'\t\tat {z} there is no solution')
                break
            # 用于最后返回
            
            if t == 0:
                assignto(x_t1,x_t)
                #x_t1 = x_t # MPC需要执行第一个时刻的放置

            assignto(x_last,x_t)
            
            if t!=0 and flag :
                findOV= findOverAndUnder(cpu_t,x_last,mem_t,M,b,y)
        # end t
        print(f'\t\tloop in k&t consuming {time()-s}')
        
        if cost < cost_min: # 选择总开销低于完全不迁移的开销且最小的
            cost_min = cost
            assignto(placement,x_t1)
            # placement = x_t1 
            print('\t\tfind placement !!!!! :',(placement ==x_t0).all() )
    # 执行完Z次随机采样后，按照最终placement矩阵的值进行调度
    # print(f'loop in z consuming {time()-zs}')
    
    del x_t
    
    return placement,cost_min