from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Flatten,
    Dropout,
    LeakyReLU,
    BatchNormalization,
    Reshape,
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.models import Model 
from tensorflow.keras.regularizers import l2 
from tensorflow.keras.layers import Attention 

class ModelBuilder:
    def __init__(self, time_steps, num_features):
        self.time_steps = time_steps
        self.num_features = num_features


    def build_model(self):
        input_layer = Input(shape=(self.time_steps, self.num_features))
        x = Reshape((self.time_steps, self.num_features, 1))(input_layer)

        # Define the input layer with the specified shape
        input_layer = Input(shape=(self.time_steps, self.num_features))
        
        # Convolutional layers to extract local patterns
        x = input_layer
        # Layer 1: Conv1D + LeakyReLU + BatchNormalization + MaxPooling
        x = Conv1D(filters=32, kernel_size=5, padding='same')(input_layer)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)  # Normalize after activation for stable training
        # x = MaxPooling1D(pool_size=3, padding='same')(x)

        # Layer 2: Conv1D + LeakyReLU + BatchNormalization + MaxPooling
        x = Conv1D(filters=32, kernel_size=5, padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=3, padding='same')(x)

        # # Layer 3: Conv1D + LeakyReLU + BatchNormalization + MaxPooling
        # x = Conv1D(filters=256, kernel_size=3, padding='same')(x)
        # x = LeakyReLU()(x)
        # x = BatchNormalization()(x)
        # x = MaxPooling1D(pool_size=5, padding='same')(x)

        # # Layer 4: Conv1D + LeakyReLU + BatchNormalization + MaxPooling
        # x = Conv1D(filters=512, kernel_size=3, padding='same')(x)
        # x = LeakyReLU()(x)
        # x = BatchNormalization()(x)
        # x = MaxPooling1D(pool_size=5, padding='same')(x)

        # Layer 5: Conv1D + LeakyReLU + BatchNormalization + MaxPooling
        x = Conv1D(filters=32, kernel_size=5, padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        # x = MaxPooling1D(pool_size=5, padding='same')(x)

        # Layer 6: Conv1D + LeakyReLU + BatchNormalization
        x = Conv1D(filters=32, kernel_size=5, padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        
        # Bidirectional LSTM layers to capture dependencies in both directions
        for _ in range(1):  # Number of LSTM layers between 2 to 3
            x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)

        # 3.Attention Layer (توجه ساده)
        attention_layer = Attention()
        context_vector = attention_layer([x, x])

        # Flatten و Dropout
        x = Flatten()(context_vector)
        x = Dropout(0.2)(x)

        # لایه خروجی با یک نورون (برای رگرسیون) با منظم‌سازی L2
        output = Dense(1, kernel_regularizer=l2(0.001))(x)

        model = Model(inputs=input_layer, outputs=output)

        optimizer = AdamW(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

        return model




