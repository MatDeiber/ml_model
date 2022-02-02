from gc import callbacks
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.models import Model, Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from keras import regularizers

reg_l1_l2 = regularizers.l1_l2(l1=0.005, l2=0.0005)


def baseline_model_v1(optimizer='adam',
                 dropout=0.1,
                 kernel_initializer='glorot_uniform'):

    model = Sequential()
    model.add(Dense(64,activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(32,activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dense(16,activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dense(1,activation='relu',kernel_initializer=kernel_initializer))

    model.compile(loss='mse',optimizer=optimizer, metrics=[keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)])
    
    return model


def baseline_model_v2(optimizer='adam',
                 dropout=0.12,
                 kernel_initializer='glorot_uniform'):

    model = Sequential()
    model.add(Dense(128,activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(64,activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dense(64,activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dense(32,activation='tanh',kernel_initializer=kernel_initializer, kernel_regularizer=reg_l1_l2))
    model.add(Dense(32,activation='tanh',kernel_initializer=kernel_initializer, kernel_regularizer=reg_l1_l2))
    model.add(Dense(1,activation='relu',kernel_initializer=kernel_initializer))

    model.compile(loss='mse',optimizer=optimizer, metrics=[keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)])
    
    return model


def baseline_model_v3(optimizer='adam',
                 dropout=0.1,
                 kernel_initializer='glorot_uniform'):

    model = Sequential()
    model.add(Dense(128,activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(64,activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(64,activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dense(1,activation='relu',kernel_initializer=kernel_initializer))

    model.compile(loss='mse',optimizer=optimizer, metrics=[keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)])
    
    return model



def get_model(model_name):
    
    es = EarlyStopping(patience=30)

    if model_name == "random_forest":

        model_params = dict(
          n_estimators=100,
          max_depth=1)

        model = RandomForestRegressor()
        model.set_params(**model_params)

        return model
    
    elif model_name == 'keras_v1':
        
        return KerasRegressor(build_fn=baseline_model_v1,verbose=1, epochs = 1000, batch_size =128, shuffle=True, validation_split=0.2, callbacks =[es])
    
        
    elif model_name == 'keras_v2':
        
        return KerasRegressor(build_fn=baseline_model_v2,verbose=1, epochs = 1000, batch_size =128, shuffle=True, validation_split=0.2, callbacks =[es])
    
    elif model_name == 'keras_v3':
        
        return KerasRegressor(build_fn=baseline_model_v3,verbose=1, epochs = 1000, batch_size =128, shuffle=True, validation_split=0.2, callbacks =[es])

