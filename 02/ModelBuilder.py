from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Bidirectional,
    LSTM,
    Dense,
    Flatten,
    Dropout,
    LeakyReLU,
    BatchNormalization,
    Reshape,
    Attention
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.models import Model 
from tensorflow.keras.regularizers import l2 

class ModelBuilder:
    def __init__(self, time_steps, num_features):
        self.time_steps = time_steps
        self.num_features = num_features

    def build_model(self):
        # Define the input layer with the specified shape
        input_layer = Input(shape=(self.time_steps, self.num_features))
        
        # Reshape input to match Conv2D requirements
        x = Reshape((self.time_steps, self.num_features, 1))(input_layer)
        
        # Convolutional layers to extract local patterns
        # Layer 1: Conv2D + LeakyReLU + BatchNormalization + MaxPooling
        x = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)  # Normalize after activation for stable training
        # x = MaxPooling2D(pool_size=(3, 1), padding='same')(x)

        # Layer 2: Conv2D + LeakyReLU + BatchNormalization + MaxPooling
        x = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(3, 3), padding='same')(x)

        # Layer 3: Conv2D + LeakyReLU + BatchNormalization + MaxPooling
        x = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        # x = MaxPooling2D(pool_size=(3, 1), padding='same')(x)

        # Layer 4: Conv2D + LeakyReLU + BatchNormalization
        x = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        
        # Bidirectional LSTM layers to capture dependencies in both directions
        x = Reshape((x.shape[1], -1))(x)  # Reshape to match LSTM input shape

        # Bidirectional LSTM layers to capture dependencies in both directions
        for _ in range(1):  # Number of LSTM layers between 2 to 3
            x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)

        # Attention Layer
        attention_layer = Attention()
        context_vector = attention_layer([x, x])

        # Flatten and Dropout
        x = Flatten()(context_vector)
        x = Dropout(0.2)(x)

        # Output layer with one neuron (for regression) with L2 regularization
        output = Dense(1, kernel_regularizer=l2(0.001))(x)

        # Define the model
        model = Model(inputs=input_layer, outputs=output)

        # Compile the model
        optimizer = AdamW(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

        return model









