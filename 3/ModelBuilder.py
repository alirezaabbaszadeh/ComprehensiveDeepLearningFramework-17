from tensorflow.keras.layers import (
    Input,
    Conv3D,
    MaxPooling3D,
    Bidirectional,
    LSTM,
    Dense,
    Flatten,
    Dropout,
    LeakyReLU,
    BatchNormalization,
    Reshape,
    Attention,
)
from tensorflow.keras.models import Model 
from tensorflow.keras.regularizers import l2 
from tensorflow.keras.optimizers import AdamW

class ModelBuilder:
    def __init__(self, time_steps, num_features):
        self.time_steps = time_steps
        self.num_features = num_features

    def build_model(self):
        # Define the input layer with the specified shape
        input_layer = Input(shape=(self.time_steps, self.num_features, 1, 1))
        
        # Convolutional layers to extract local patterns
        x = input_layer
        # Layer 1: Conv3D + LeakyReLU + BatchNormalization + MaxPooling
        x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)  # Normalize after activation for stable training
        # x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)

        # Layer 2: Conv3D + LeakyReLU + BatchNormalization + MaxPooling
        x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)

        # Layer 3: Conv3D + LeakyReLU + BatchNormalization + MaxPooling
        x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        # x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)

        # Layer 4: Conv3D + LeakyReLU + BatchNormalization
        x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        
        # Reshape for LSTM input
        x = Reshape((self.time_steps, -1))(x)

        # Bidirectional LSTM layers to capture dependencies in both directions
        for _ in range(1):  # Number of LSTM layers between 2 to 3
            x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)

        # Attention Layer
        attention_layer = Attention() 
        context_vector = attention_layer([x, x])

        # Flatten and Dropout
        x = Flatten()(context_vector)
        x = Dropout(0.2)(x)

        # Output layer with a single neuron (for regression) with L2 regularization
        output = Dense(1, kernel_regularizer=l2(0.001))(x)

        model = Model(inputs=input_layer, outputs=output)

        # Compile the model
        optimizer = AdamW(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

        return model
