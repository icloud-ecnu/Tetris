import pmdarima as pm


class Container(object):
    def __init__(self, container_config):
        self.id = container_config["id"]
        self.name = container_config["containerName"]
        self.mac_id = container_config["node_id"]
        self.cpu = container_config["cpu"]
        self.mem = container_config["mem"]
        self.memlist = container_config["memory_curve"].copy()
        self.cpulist = container_config["cpu_curve"].copy()
        self.predictCpuList = []
        self.predictMemList = []
        self.predicts = {}
        self.node = None
        self.lastmacineId = -1

    def attach(self, node):
        self.node = node
        self.mac_id = node.id

    def predict(self, clock, w, flag=False):
        if flag:
            return self.predicts
        predicts = self.predicts
        model = pm.auto_arima(self.cpulist, start_p=0, start_q=0,
                              information_criterion='aic',
                              test='adf',       # use adftest to find optimal 'd'
                              max_p=3, max_q=3,  # maximum p and q
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
        model = model.fit(self.memlist)
        predict_mem = self.memlist + self.predictMemList[-1]
        self.cpu = predict_cpu[0]
        self.mem = predict_mem[0]
        predicts[clock] = {"cpu": predict_cpu, "mem": predict_mem}
        return predicts
