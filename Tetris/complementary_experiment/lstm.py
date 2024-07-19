from tensorflow.python.keras.models import Sequential
import numpy as np
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping

# 返回lstm模型
def getLstmModel(data, trainRatio, window):
    data = np.array(data[0:int(len(data)*trainRatio)]).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # print(data_scaled)
    X_train = []
    y_train = []
    for i in range(window, len(data_scaled)):
        X_train.append(data_scaled[i-window:i, 0])
        y_train.append(data_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    print(f'x_train\'s shape is {X_train.shape}\n')
    print(f'X_train.shape[1] is {X_train.shape[1]}')

    model = Sequential()
    model.add(LSTM(units=50, activation='softsign', return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, activation='softsign', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, activation='softsign', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, activation='softsign', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    es_callback = EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
    model.fit(X_train, y_train, epochs=50, batch_size=4, validation_split=0.25, callbacks=[es_callback], verbose=0)

    return model, scaler

def getLstmModel1(data, trainRatio, window):
    data = np.array(data[:int(len(data) * trainRatio)]).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X_train = []
    y_train = []
    for i in range(window, len(data_scaled)):
        X_train.append(data_scaled[i - window:i, 0])
        y_train.append(data_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    if X_train.size == 0 or y_train.size == 0:
        raise ValueError("Insufficient data for training. Increase the size of your dataset or reduce the window size.")
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # 确保 input_shape 是一个普通的 Python 元组
    input_shape = (X_train.shape[1], 1)

    units=30
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    es_callback = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=30, batch_size=4, validation_split=0.2, callbacks=[es_callback], verbose=0)

    return model, scaler

def lstm_forecast_with_model(model, scaler, data, window, n_steps):
    data = np.array(data).reshape(-1, 1)
    data_scaled = scaler.transform(data)
    
    # x_input = data_scaled[-window:].reshape(1, -1, 1)
    x_input = data_scaled[-window:].reshape(1, window, 1)
    forecast = []
    for _ in range(n_steps):
        yhat = model.predict(x_input, verbose=0)
        forecast.append(yhat[0, 0])
        x_input = np.append(x_input[:, 1:, :], yhat.reshape(1, 1, 1), axis=1)
    
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    return forecast

