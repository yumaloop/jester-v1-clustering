import keras

def ConvLSTMAutoEncoder(input_shape=(50, 48, 48, 3)):
    """
    keras.layers.LSTM(filters, kernel_size, 
        activation='tanh', 
        recurrent_activation='hard_sigmoid', 
        use_bias=True, 
        kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)

    keras.layers.ConvLSTM2D():
        Arguments:
        ==========
        filters: int,
            出力空間の次元（つまり畳み込みにおける出力フィルタの数）．
        kernel_size: 整数かn個の整数からなるタプル/リスト
            n次元の畳み込みウィンドウを指定．
        strides: int or tuple or list
            畳み込みのストライドをそれぞれ指定．strides value != 1とするとdilation_rate value != 1と指定できません．
        padding: string
            "valid" or "same"
        data_format: string
            "channels_last" or "channels_first"
        dilation_rate: int or tuple or list
            DeConvの膨張率
        activation: 



    (samples, time, filters, output_row, output_col)
        return_sequences: 真理値．出力系列の最後の出力を返すか，完全な系列を返すか．

    """
    model = keras.models.Sequential(name="ConvLSTMAutoEncoder")

    # Encoder: ConvLSTM
    model.add(keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(keras.layers.Conv3D(filters=3, kernel_size=(3,3,3), padding='same'))

    # Decoder: ConvLSTM
    model.add(keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.UpSampling3D(size=(1, 2, 2)))
    model.add(keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.UpSampling3D(size=(1, 2, 2)))
    model.add(keras.layers.Conv3D(filters=3, kernel_size=(3,3,3), padding='same'))
    
    return model
