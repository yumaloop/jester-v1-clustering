import keras

def ConvLSTMAutoEncoder(input_shape=(50, 48, 48, 3)):
    """
    keras.layers.ConvLSTM2D():
        Args:
        =====
        filters: Integer, 
            出力空間の次元（つまり畳み込みにおける出力フィルタの数）．
            the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, 
            n次元の畳み込みウィンドウを指定．
            specifying the dimensions of the convolution window.
        strides: An integer or tuple/list of n integers,
            畳み込みのストライドをそれぞれ指定．strides value != 1とするとdilation_rate value != 1と指定できません．
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `"channels_last"` (default) or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            - `"channels_last"` corresponds to inputs with shape `(batch, time, ..., channels)`
            - `"channels_first"` corresponds to inputs with shape `(batch, time, channels, ...)`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: An integer or tuple/list of n integers, 
            DeConvolution時点の膨張率
            specifying the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: 
            Activation function to use tanh is applied by default.
        recurrent_activation: 
            Activation function to use for the recurrent step
        use_bias: Boolean, 
            whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel` weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: 
            Initializer for the bias vector
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Use in combination with `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al. (2015)](
            http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
        recurrent_regularizer: 
            Regularizer function applied to the `recurrent_kernel` weights matrix
        bias_regularizer: 
            Regularizer function applied to the bias vector
        activity_regularizer: 
            Regularizer function applied to the output of the layer (its "activation").
        kernel_constraint: 
            Constraint function applied to the `kernel` weights matrix
        recurrent_constraint: 
            Constraint function applied to the `recurrent_kernel` weights matrix
        bias_constraint: 
            Constraint function applied to the bias vector
        return_sequences: Boolean. 
            真理値．出力系列の最後の出力を返すか，完全な系列を返すか．
            Whether to return the last output in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards.
        stateful: Boolean (default False).
            If True, the last state for each sample at index i in a batch 
            will be used as initial state for the sample of index i in the following batch.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.

    rf.) https://github.com/keras-team/keras/blob/master/keras/layers/convolutional_recurrent.py#L773
    """
    # -------------------------------
    # Encoder: (ConvLSTM AutoEncoder)
    # -------------------------------
    input_seq  = keras.layers.Input(shape=input_shape)
    input_seq_rev = keras.layers.Lambda(lambda x: keras.backend.reverse(x, axes=1))(input_seq)
    input_seq_rev = keras.layers.Lambda(lambda x: x[:, :-1])(input_seq_rev) 

    h1_seq = keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(input_seq)
    h1_seq = keras.layers.BatchNormalization()(h1_seq)
    # h1_seq = keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(h1_seq)

    h2_seq = keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(h1_seq)
    h2_seq = keras.layers.BatchNormalization()(h2_seq)
    # h2_seq = keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(h2_seq)

    latent = keras.layers.ConvLSTM2D(filters=3, kernel_size=(3, 3), padding='same', return_sequences=False)(h2_seq)
    latent = keras.layers.Lambda(lambda x: keras.backend.expand_dims(x, axis=1))(latent)

    # -------------------------------
    # Decoder: (ConvLSTM AutoEncoder)
    # -------------------------------
    input_latent = keras.layers.concatenate([latent, input_seq_rev], axis=1)
    h3_seq = keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(input_latent)
    h3_seq = keras.layers.BatchNormalization()(h3_seq)
    # h3_seq = keras.layers.UpSampling3D(size=(1, 2, 2))(h3_seq)

    h4_seq = keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(h3_seq)
    h4_seq = keras.layers.BatchNormalization()(h4_seq)
    # h4_seq = keras.layers.UpSampling3D(size=(1, 2, 2))(h4_seq)

    h5_seq = keras.layers.ConvLSTM2D(filters=3, kernel_size=(3, 3), padding='same', return_sequences=True)(h4_seq)
    h5_seq = keras.layers.BatchNormalization()(h5_seq)
    # h4_seq = keras.layers.UpSampling3D(size=(1, 2, 2))(h4_seq)

    output_seq = keras.layers.Lambda(lambda x: keras.backend.reverse(x, axes=1))(h5_seq)
    # output_seq = keras.layers.Conv3D(filters=3, kernel_size=(3,3,3), padding='same')(output_seq)
    
    model = keras.models.Model(input=input_seq, output=output_seq, name="ConvLSTMAutoEncoder")
    return model
