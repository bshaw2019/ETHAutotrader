from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers
from keras.models import model_from_json
import keras
import os.path

def build_model(layers):
    d = 0.2
    model = Sequential()
    model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(16, kernel_initializer="uniform"))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(layers[2], kernel_initializer="uniform"))
    model.add(LeakyReLU(alpha=0.3))
    #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer="rmsprop", metrics=['accuracy'])
    return model

def save_model(model):
    dir_path = os.path.dirname(os.path.abspath(__file__))    
    json_string = model.to_json()
    open(os.path.join(dir_path, 'model.json'), 'w').write(json_string)
    model.save_weights(os.path.join(dir_path, 'weights.h5'), overwrite=True)
    
def load_model():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model = model_from_json(open(os.path.join(dir_path, 'model.json')).read())
    model.load_weights(os.path.join(dir_path, 'weights.h5'))
    rms = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    #model.compile(optimizer=rms, loss = 'mse')
    model.compile(loss='mse', optimizer="rmsprop", metrics=['accuracy'])
    return model
