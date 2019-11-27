import keras
import keras.backend as K

def PredNet(input_shape=(50, 64, 64, 3)):
    # Input
    x_0 = keras.layers.Input(shape=input_shape,    dtype='float32', name='x_input')
    e_0 = keras.layers.Input(shape=(1, 64, 64, 6), dtype='float32', name='e_input')
    r_0 = keras.layers.Input(shape=(64, 64, 3),    dtype='float32', name='r_input')

    for t in range(T):

        if t == 0:
            for l in reversed(range(L)):
                if l == L:
                    R, state_h, state_c = keras.layers.ConvLSTM2D(3, (3, 3), padding='same', return_sequences=False, return_state=True, name='R')(e_0)
                else:
                    R, state_h, state_c = keras.layers.ConvLSTM2D(3, (3, 3), padding='same', return_sequences=False, return_state=True, name='R')(E)
            for l in range(L):
                if l == 0:
                    Ahat = keras.layers.Conv2D(3, (3, 3), padding='same')(r_0)
                    Ahat = kersa.layers.Activation()(Ahat)
                else:
                    Ahat = keras.layers.Conv2D(3, (3, 3), padding='same')(R)
                    Ahat = kersa.layers.Activation()(Ahat)
        else:
            for l in reversed(range(L)):
                if l == L:
                    R, state_h, state_c = keras.layers.ConvLSTM2D(3, (3, 3), padding='same', return_sequences=False, return_state=True, name='R')(e_input)
                else:
                    keras.layers.Conv2D(3, (3, 3))(x_input)
                    A = keras.layers.Activation('relu', name='ReLU')()

        



    # A unit
    A = keras.layers.Conv2D(3, (3, 3), padding='same', name='A')(x_input)

    # R unit 
    e_input = keras.layers.Input(shape=(1, 64, 64, 6), dtype='float32', name='e_input')
    r_input = keras.layers.Input(shape=(64, 64, 3),    dtype='float32', name='r_input')
    R, state_h, state_c = keras.layers.ConvLSTM2D(3, (3, 3), padding='same', return_sequences=False, return_state=True, name='R')(e_input)

    # A hat unit
    Ahat = keras.layers.Conv2D(3, (3, 3), padding='same', name='Ahat')(R)

    # E unit : pixcel layer (l=0) 
    e0 = keras.layers.Subtract()([x_input, Ahat])
    e0 = keras.layers.Activation('relu', name='ReLU_0')(e0)
    e1 = keras.layers.Subtract()([Ahat, x_input])
    e1 = keras.layers.Activation('relu', name='ReLU_1')(e1)
    E =  keras.layers.Concatenate(axis=-1)([e0, e1])

    # Model 
    predict_m = keras.models.Model(input=[e_input], output=[Ahat])
    predict_m.compile(optimizer='rmsprop', loss='mean_squared_error')

    model = keras.models.Model(input=[x_input,e_input], output=[E])
    model.compile(optimizer='rmsprop', loss='mean_squared_error')


def PredNet_Original():
    """
    Input
        X:    (batch_size, T, 1, h, w, c)
    Output
        A     (batch_size, T, 1, h, w, c)
    State
        R:    (batch_size, T, L, h, w, c) init all elems as 0
        E:    (batch_size, T, L, h, w, c) init all elems as 0
        A:    (batch_size, T, L, h, w, c) 
        Ahat: (batch_size, T, L, h, w, c)
    """

    # Input
    x_input = keras.layers.Input(shape=(50, 1,  64, 64, 3), dtype='float32', name='x_input')
    e_input = keras.layers.Input(shape=(50, 10, 64, 64, 6), dtype='float32', name='e_input')
    r_input = keras.layers.Input(shape=(50, 10, 64, 64, 3), dtype='float32', name='r_input')

    # A unit
    A = keras.layers.wrappers.TimeDistributed(keras.layers.Conv2D(3, (3, 3), padding='same', name='A'))(e_input)

    # 
    x0_input = K.gather(x_input, 0)



    # A = keras.layers.Conv2D(3, (3, 3), padding='same', name='A')(x_input)

    # R unit 
    e_input = keras.layers.Input(shape=(1,  64, 64, 6), dtype='float32', name='e_input')
    r_input = keras.layers.Input(shape=(64, 64, 3),     dtype='float32', name='r_input')
    R, state_h, state_c = keras.layers.ConvLSTM2D(3, (3, 3), padding='same', return_sequences=False, return_state=True, name='R')(e_input)

    # A hat unit

    keras.layers.wrappers.TimeDistributed(keras.layser.Conv2D(3, (3, 3), padding='same', name='Ahat'))(R)
    Ahat = keras.layers.Conv2D(3, (3, 3), padding='same', name='Ahat')(R)

    # E unit : pixcel layer (l=0) 
    e0 = keras.layers.Subtract()([x_input, Ahat])
    e0 = keras.layers.Activation('relu', name='ReLU_0')(e0)
    e1 = keras.layers.Subtract()([Ahat, x_input])
    e1 = keras.layers.Activation('relu', name='ReLU_1')(e1)
    E =  keras.layers.Concatenate(axis=-1)([e0, e1])

    # Model 
    predict_m = keras.models.Model(input=[e_input], output=[Ahat])
    predict_m.compile(optimizer='rmsprop', loss='mean_squared_error')

    model = keras.models.Model(input=[x_input,e_input], output=[E])
    model.compile(optimizer='rmsprop', loss='mean_squared_error')

    return predict_m, model


def PredNet_Original():

    # Input
    x_input = keras.layers.Input(shape=(50, 1, 64, 64, 3), dtype='float32', name='x_input')
    e_input = keras.layers.Input(shape=(50, 1, 64, 64, 6), dtype='float32', name='e_input')
    r_input = keras.layers.Input(shape=(50, 1, 64, 64, 3), dtype='float32', name='r_input')
    
    # 
    A = keras.layers.Conv2D(3, (3, 3), padding='same', name='A')(x_input)

    # R unit 
    e_input = keras.layers.Input(shape=(1,  64, 64, 6), dtype='float32', name='e_input')
    r_input = keras.layers.Input(shape=(64, 64, 3),     dtype='float32', name='r_input')
    R, state_h, state_c = keras.layers.ConvLSTM2D(3, (3, 3), padding='same', return_sequences=False, return_state=True, name='R')(e_input)

    # A hat unit

    keras.layers.wrappers.TimeDistributed(keras.layser.Conv2D(3, (3, 3), padding='same', name='Ahat'))(R)
    Ahat = keras.layers.Conv2D(3, (3, 3), padding='same', name='Ahat')(R)

    # E unit : pixcel layer (l=0) 
    e0 = keras.layers.Subtract()([x_input, Ahat])
    e0 = keras.layers.Activation('relu', name='ReLU_0')(e0)
    e1 = keras.layers.Subtract()([Ahat, x_input])
    e1 = keras.layers.Activation('relu', name='ReLU_1')(e1)
    E =  keras.layers.Concatenate(axis=-1)([e0, e1])

    # Model 
    predict_m = keras.models.Model(input=[e_input], output=[Ahat])
    predict_m.compile(optimizer='rmsprop', loss='mean_squared_error')

    model = keras.models.Model(input=[x_input,e_input], output=[E])
    model.compile(optimizer='rmsprop', loss='mean_squared_error')

