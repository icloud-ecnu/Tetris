
import numpy as np
def CostOfLoadBalance(cpu_t, mem_t, x_t, b):
    """计算负载均衡开销算法

    Args:
        cpu_t (numpy.ndarray): t时刻各VM的cpu需求量
        mem_t (numpy.ndarray): t时刻各VM的mem需求量
        x_t (numpy.ndarray): t时刻VM放置情况
        b (float): 权重

    Returns:
        float: t时刻的负载均衡开销
    """
    CPU_t = ResourceUsage(cpu_t, x_t)
    MEM_t = ResourceUsage(mem_t, x_t)
    cost_cpu = np.sum(np.square(CPU_t)) # 每个元素平方后求和（参照化简后公式）
    cost_mem = np.sum(np.square(MEM_t))
    cost_bal = cost_cpu + b * cost_mem
    
    return cost_bal


def CostOfMigration(x_last, x_now, mem_now):
    """计算迁移开销算法
    Args:
        x_last (numpy.ndarray): t-1时刻VM放置情况 N*M (vm N个, pm M个)
        x_now (numpy.ndarray): t时刻的VM放置情况 N*M (vm N个, pm M个)
        mem_now (numpy.ndarray): t时刻的mem需求 1*N

    Returns:
        float: t时刻的迁移开销
    """
    mig = ~(x_last == x_now).all(axis = 1) # 表示是否迁移的矩阵，即[True,False,False]表示VM_1迁移了
    cost_mig = sum(mig * mem_now) # 当前时刻每VM的单位迁移开销为其此刻mem占用
    
    return cost_mig

# 计算每台机器上资源使用量
# 输入为t时刻各VM的cpu需求量以及VM放置情况
# 输出为t时刻各机器的资源使用总量
def ResourceUsage(cpuormem_t, x_t):
    """计算每台机器上资源使用量

    Args:
        cpuormem_t (numpy.ndarray): t时刻各VM的cpu或mem需求量
        x_t (numpy.ndarray): t时刻VM放置情况 N*M (vm N个, pm M个)

    Returns:
        float: t时刻各机器的资源使用总量
    """
    x_cput = x_t.copy().T
    CPU_t = np.matmul(x_cput,cpuormem_t)
    
    return CPU_t




def isAllUnderLoad(x_t, cpu_t, mem_t, CPU_MAX, MEM_MAX):
    """判断所有机器是否过载

    Args:
        x_t0 (numpy.ndarray): t0时刻VM部署情况 N*M (vm N个, pm M个)
        cpu_t0 (numpy.ndarray): N个VM当前时刻的cpu需求量
        mem_t0 (numpy.ndarray): N个VM当前时刻的cpu需求量
        CPU_MAX (int): cpu资源使用上限
        MEM_MAX (int): mem资源使用上限

    Returns:
        bool: Ture or Not
    """
    CPU_t = ResourceUsage(cpu_t, x_t)
    MEM_t = ResourceUsage(mem_t, x_t)
    print("---- test sandpiper ----",CPU_t,MEM_t)
    is_cpu = (CPU_t < CPU_MAX).all()
    is_mem = (MEM_t < MEM_MAX).all()
    is_all = is_cpu and is_mem # 所以资源都不过载才返回True
    
    return is_all

def CostOfSingleMachineLoadBalance(x_t, pm, cpu_t, mem_t, b):
    """_summary_

    Args:
        x_t (numpy.ndarray): t时刻VM部署情况 N*M (vm N个, pm M个)
        pm (int): 定机器的id,0<pm<M
        cpu_t (numpy.ndarray): N个VM当前时刻的cpu需求量
        mem_t (numpy.ndarray): N个VM当前时刻的cpu需求量
        b (float): 权重

    Returns:
        float: 负载均衡指标
    """
    vm = np.where(x_t[:, pm] == 1)[0] # 在指定机器上的VM索引
    cpu = cpu_t[vm] # 对应VM的cpu
    mem = mem_t[vm]
    pm_cost = 0
    for i in range(len(vm)-1): # 每台VM与其他VM两两相乘
        c = cpu[i] * np.sum(cpu[i+1:])
        m = mem[i] * np.sum(mem[i+1:])
        pm_cost += c + b * m
    return pm_cost