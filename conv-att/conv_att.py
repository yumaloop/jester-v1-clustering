import numpy as np
import keras
import keras.backend as K

class CNNAttention( ):
    def __init__(self, T=40, L=10, img_shape=(64,64,3)):
        self.T = T
        self.L = L
        self.img_h = img_shape[0]
        self.img_w = img_shape[1]
        self.img_c = img_shape[2]
        
        self.x_shape = (self.T, 1,      self.img_h, self.img_w, self.img_c    )
        self.e_shape = (self.T, self.L, self.img_h, self.img_w, self.img_c * 2)
        self.r_shape = (self.T, self.L, self.img_h, self.img_w, self.img_c    )
        
    def build_train(self):
        # Input Lauer
        # T: 50, L: 10, h:64, w:64, c:3
        X_input = keras.layers.Input(shape=self.x_shape, dtype='float32', name='x_input')
        E_input = keras.layers.Input(shape=self.e_shape, dtype='float32', name='e_input')
        R_input = keras.layers.Input(shape=self.r_shape, dtype='float32', name='r_input')

        for t in range(self.T):

            """ === E unit === """
            if t == 0:
                E = keras.layers.Lambda(lambda x: x)(E_input)
                def E_to_Et(E):
                    # E_t: (None, L, h, w, 2c)
                    E_t = K.permute_dimensions(E, [1, 0, 2, 3, 4, 5])
                    E_t = K.gather(E_t, [t])
                    E_t = K.permute_dimensions(E_t, [1, 0, 2, 3, 4, 5])
                    E_t = K.squeeze(E_t, axis=1)
                    return E_t
                E_t = keras.layers.Lambda(E_to_Et)(E)

            """ === R unit === """
            if t == 0:
                def R_to_Rt(R_input):
                    R_t = K.permute_dimensions(R_input, [1, 0, 2, 3, 4, 5])
                    R_t = K.gather(R_t, [t])
                    R_t = K.permute_dimensions(R_t, [1, 0, 2, 3, 4, 5])
                    R_t = K.squeeze(R_t, axis=1)
                    return R_t
                R_t = keras.layers.Lambda(R_to_Rt)(R_input)
            else:
                # R_t: (None, L, h, w, 3)
                E_t_rev = keras.layers.Lambda(lambda x: K.reverse(x, axes=1))(E_t)
                R_t, state_h_t, state_c_t = keras.layers.ConvLSTM2D(3, (3, 3), 
                                                                    padding='same', 
                                                                    activation='tanh', 
                                                                    return_sequences=True, 
                                                                    return_state=True)(E_t_rev)
            

            for l in range(self.L):  
                """ === R_tl === """
                def Rt_to_Rtl(R_t):
                    R_tl = K.permute_dimensions(R_t, [1, 0, 2, 3, 4])
                    R_tl = K.gather(R_tl, [l])
                    R_tl = K.permute_dimensions(R_tl, [1, 0, 2, 3, 4])
                    R_tl = K.squeeze(R_tl, axis=1)
                    return R_tl
                # R_tl: (None, h, w, 3)
                R_tl = keras.layers.Lambda(Rt_to_Rtl)(R_t)                
                print("t:",t,"l:",l,"R_tl",R_tl)
                
                """ === Ahat_tl === """
                # Ahat_tl: (None, h, w, 3)
                Ahat_tl = keras.layers.Conv2D(3, (3, 3), padding='same')(R_tl)
                Ahat_tl = keras.layers.Activation('relu')(Ahat_tl)
                print("t",t,"l,",l,"Ahat_tl",Ahat_tl)

                if l == 0:                    
                    def X_to_Atl(x_input):
                        # (None, T, 1, h, w, c) --> (?, h, w, c)
                        A_tl = K.squeeze(x_input, axis=2) # (None, T, 64, 64, 3)
                        A_tl = K.permute_dimensions(A_tl, [1, 0, 2, 3, 4]) # (T, None, 64, 64, 3)
                        A_tl = K.gather(A_tl, [t]) # (1, None, 64, 64, 3)
                        A_tl = K.squeeze(A_tl, axis=0) # (None, 64, 64, 3)
                        return A_tl

                    # A_tl: (None, h, w, 3)
                    A_tl = keras.layers.Lambda(X_to_Atl)(X_input)

                # E_tl: (None, h, w, c)
                err0_tl = keras.layers.Subtract()([A_tl, Ahat_tl])
                err0_tl = keras.layers.Activation('relu')(err0_tl)
                err1_tl = keras.layers.Subtract()([Ahat_tl, A_tl])
                err1_tl = keras.layers.Activation('relu')(err1_tl)
                E_tl    = keras.layers.Concatenate(axis=-1)([err0_tl, err1_tl])

                # A_tl: (None, h, w, 3)
                if l < self.L-1:
                    def Et_to_Etl(E_t):
                        E_tl = K.permute_dimensions(E_t, [1, 0, 2, 3, 4])
                        E_tl = K.gather(E_tl, [l])
                        E_tl = K.permute_dimensions(E_tl, [1, 0, 2, 3, 4])
                        E_tl = K.squeeze(E_tl, axis=1)
                        return E_tl

                    # E_tl: (None, h, w, c)
                    E_tl = keras.layers.Lambda(Et_to_Etl)(E_t)
                    # A_tl
                    A_tl = keras.layers.Conv2D(3, (3, 3), padding='same')(E_tl)

                if l == 0:
                    # E_l: (None, 1, h, w, c)
                    E_t = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(E_tl)
                else:
                    # E_tl_: (None, 1, h, w, c)
                    E_tl_ = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(E_tl)
                    # E_l: (None, T, h, w, c)
                    E_t = keras.layers.Concatenate(axis=1)([E_t, E_tl_])
            if t == 0:
                # E: (None, 1, T, h, w, c)
                E = keras.layers.Lambda(lambda x : K.expand_dims(x, axis=1))(E_t)
            else:
                # E_l_: (None, 1, T, h, w, c)
                E_t_ = keras.layers.Lambda(lambda x : K.expand_dims(x, axis=1))(E_t)
                # E: (None, L, T, h, w, c)
                E = keras.layers.Concatenate(axis=1)([E, E_t_])
        print("t:", t, "l:", l, "E:", E)

        # print(E) # 50, 10, 64, 64, 6
        model_train = keras.models.Model(inputs=[X_input, E_input, R_input], outputs=[E])
        return model_train
