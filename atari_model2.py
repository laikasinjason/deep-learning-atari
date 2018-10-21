class Atari_Model2:
    # tensorflow backend
    def __create_model(self, n_actions, alpha = 0.00025):
        ATARI_SHAPE = (84, 84, 4)

        frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
#         actions_input = keras.layers.Input((self.n_actions,), name='filter')

        conv_1 = keras.layers.convolutional.Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu'
        )(keras.layers.Lambda(lambda x: x / 255.0)(frames_input))
        conv_2 = keras.layers.convolutional.Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu')(conv_1)
        conv_3 = keras.layers.convolutional.Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu')(conv_2)
        conv_flattened = keras.layers.core.Flatten()(conv_3)
        hidden = keras.layers.Dense(512, activation='relu')(conv_flattened)
        output = keras.layers.Dense(self.n_actions)(hidden)
#         filtered_output = keras.layers.multiply([output, actions_input])
        
#         model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
        model = keras.models.Model(input=[frames_input], output=output)
        optimizer = optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        model.compile(optimizer, loss=huber_loss)
        
        return model
        
    def __init__(self, n_actions):
        self.n_actions = n_actions

        self.model = self.__create_model(self.n_actions)
        self.target_model = self.__create_model(self.n_actions)
        
    def copy_model(self):
        self.target_model.set_weights(self.model.get_weights())