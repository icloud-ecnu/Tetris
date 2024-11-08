import pmdarima as pm       
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
tf.config.run_functions_eagerly(True)

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
    
    import tensorflow as tf

    def predict(self, clock, w, flag=False):
        if flag:
            return self.predicts

        predicts = self.predicts
        model = pm.auto_arima(self.cpulist, start_p=0, start_q=0,
                            information_criterion='aic',
                            test='adf',       # use adftest to find optimal 'd'
                            max_p=3, max_q=3, # maximum p and q
                            d=0,               # let model determine 'd'
                            seasonal=True,     # No Seasonality
                            trace=False,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)

        data = np.array(self.memlist).reshape(-1, 1)
        generator = TimeseriesGenerator(data, data, length=w, batch_size=1)
        generator_data = np.array([x[0] for x in generator])  # 将数据部分提取成 numpy 数组
        generator_target = np.array([x[1] for x in generator])  # 将目标部分提取成 numpy 数组

        lstm_model = Sequential()
        lstm_model.add(LSTM(50, activation='relu', input_shape=(w, 1)))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mse')

        # lstm_model.fit(generator, epochs=10, verbose=0)
        lstm_model.fit(generator_data, generator_target, epochs=10, verbose=0)

        mem_predictions = []
        last_sequence = data[-w:]  # 获取最后 w 个数据，作为初始预测窗口

        for _ in range(w):
            prediction = lstm_model.predict(last_sequence.reshape(1, w, 1))
            mem_predictions.append(prediction[0, 0])

            # 将 last_sequence 处理为 TensorFlow 张量，并拼接 prediction
            last_sequence = tf.concat([last_sequence[1:], tf.constant(prediction.flatten())], axis=0)

        self.predictCpuList.append(model.predict(n_periods=w).tolist())
        self.predictMemList.append(mem_predictions)

        predict_cpu = self.cpulist + self.predictCpuList[-1]
        predict_mem = self.memlist + self.predictMemList[-1]

        self.cpu = predict_cpu[0]
        self.mem = predict_mem[0]

        predicts[clock] = {"cpu": predict_cpu, "mem": predict_mem}

        return predicts

