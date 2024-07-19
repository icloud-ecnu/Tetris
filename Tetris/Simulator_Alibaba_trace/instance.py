import pmdarima as pm       

class InstanceConfig(object):
    def __init__(self, machine_id,instance_id, cpu, memory, disk, cpu_curve=None, memory_curve=None):
        self.id = instance_id
        self.machine_id = machine_id
        self.cpu = cpu 
        self.mem = 0
        self.memory = memory
        self.disk = disk
        self.cpu_curve = cpu_curve
        self.memory_curve = memory_curve

        

class Instance(object):
    def __init__(self, instance_config:InstanceConfig):
        self.id = instance_config.id
        self.mac_id = instance_config.machine_id
        self.cpu = instance_config.cpu
        self.mem = instance_config.memory
        self.memlist = instance_config.memory_curve.copy()
        self.disk = instance_config.disk
        self.cpulist = instance_config.cpu_curve.copy()
        self.predicts = {}
        self.machine = None
        self.lastmacineId = -1
        self.predictCpuList = []  # 初始化predictCpuList
        self.predictMemList = []  # 初始化predictMemList

    def attach(self, machine):
        self.machine = machine
    
    def predict(self,clock,w,flag=False):
        if flag:
            return self.predicts
        
        predicts = self.predicts
        model = pm.auto_arima(self.cpulist, start_p=0, start_q=0,
                      information_criterion='aic',
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                                  # frequency of series
                      d=0,           # let model determine 'd'
                      seasonal=True,   # No Seasonality
                      trace=False,
                      error_action='ignore', 
                      suppress_warnings=True,
                      stepwise=True)
        
        self.predictCpuList.append(model.predict(n_periods=w).tolist())
        self.predictMemList.append(model.predict(n_periods=w).tolist())
        
        predict_cpu = self.cpulist + self.predictCpuList[-1]
        # model = model.fit(self.memlist) # --------------- occur error
        predict_mem = self.memlist + self.predictMemList[-1]
        
        self.cpu = predict_cpu[0]
        self.mem = predict_mem[0]
        
        predicts[clock] = {"cpu":predict_cpu,"mem":predict_mem}
        
        return predicts
