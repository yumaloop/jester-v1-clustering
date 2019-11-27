import keras

def ConvLSTMSeq2SeqAutoEncoder(input_shape=(None, 48, 48, 3)):
    # -------------------------------
    # Encoder: (ConvLSTM AutoEncoder)
    # -------------------------------
    input_seq  = keras.layers.Input(shape=input_shape)

    h1_seq, state1_h, state1_c  = keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_state=True, return_sequences=True)(input_seq)
    h1_seq = keras.layers.BatchNormalization()(h1_seq)

    h2_seq, state2_h, state2_c = keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_state=True, return_sequences=True)(h1_seq)
    h2_seq = keras.layers.BatchNormalization()(h2_seq)

    latent, state3_h, state3_c = keras.layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_state=True, return_sequences=True)(h2_seq)
    latent = keras.layers.BatchNormalization()(latent)

    # -------------------------------
    # Decoder: (ConvLSTM AutoEncoder)
    # -------------------------------
    h3_seq, state3_h, state3_c = keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_state=True, return_sequences=True)(latent)
    h3_seq = keras.layers.BatchNormalization()(h3_seq)

    h4_seq, state4_h, state4_c = keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_state=True, return_sequences=True)(h3_seq)
    h4_seq = keras.layers.BatchNormalization()(h4_seq)
    
    h5_seq, state5_h, state5_c = keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_state=True, return_sequences=True)(h4_seq)
    h5_seq = keras.layers.BatchNormalization()(h5_seq)
    
    output_seq = h5_seq
    
    model = keras.models.Model(input=input_seq, output=output_seq, name="ConvLSTMSeq2SeqAutoEncoder")
    return model


