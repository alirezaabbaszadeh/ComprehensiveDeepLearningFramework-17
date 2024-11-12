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
    MultiHeadAttention,
    Add,
    AdditiveAttention
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
import tensorflow as tf

class ConvLayerConfig:
    def __init__(self, block_configs):
        self.block_configs = block_configs

class ModelBuilder:
    def __init__(self, time_steps, num_features, block_configs, num_heads=4, key_dim=128):
        self.time_steps = time_steps
        self.num_features = num_features
        self.block_configs = block_configs
        self.num_heads = num_heads
        self.key_dim = key_dim

    def residual_block(self, x, filters, kernel_size, pool_size, block_num):
        shortcut = x
        x = Conv3D(filters=filters, kernel_size=kernel_size, padding='same', name=f'conv1_{block_num}')(x)
        x = LeakyReLU(name=f'leaky_relu1_{block_num}')(x)
        x = BatchNormalization(name=f'bn1_{block_num}')(x)
        
        # Adding Multi-Head Attention after the first convolution
        attention_layer_1 = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, name=f'attention1_{block_num}')
        x = attention_layer_1(x, x)
        
        x = Conv3D(filters=filters, kernel_size=kernel_size, padding='same', name=f'conv2_{block_num}')(x)
        x = BatchNormalization(name=f'bn2_{block_num}')(x)
        
        # Adding Multi-Head Attention after the second convolution
        attention_layer_2 = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, name=f'attention2_{block_num}')
        x = attention_layer_2(x, x)
        
        transformed_shortcut = shortcut
        if shortcut.shape[-1] != filters:
            transformed_shortcut = Conv3D(filters=filters, kernel_size=1, padding='same', name=f'shortcut_conv_{block_num}')(shortcut)
            transformed_shortcut = BatchNormalization(name=f'shortcut_bn_{block_num}')(transformed_shortcut)
              
        x = Add(name=f'add_{block_num}')([transformed_shortcut, x])
        x = LeakyReLU(name=f'leaky_relu2_{block_num}')(x)
        
        if pool_size is not None:
            x = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size), padding='same', name=f'max_pool_{block_num}')(x)
        
        return x

    def build_model(self):
        # Define the input layer with a more explicit shape
        input_layer = Input(shape=(self.time_steps, self.num_features, 1, 1))
        
        # Convolutional layers to extract local patterns
        x = input_layer
        for i, block in enumerate(self.block_configs):
            filters = block.get('filters', 32)
            kernel_size = block.get('kernel_size', (3, 3, 3))
            pool_size = block.get('pool_size', None)
            x = self.residual_block(x, filters, kernel_size, pool_size, i)
        
        # Reshape for LSTM input without using TensorFlow functions directly on KerasTensor
        x = Flatten()(x)
        new_shape = x.shape[-1] // self.time_steps
        if new_shape * self.time_steps != x.shape[-1]:
            raise ValueError("The dimensions are not compatible for reshaping. Please check the input shape.")
        x = Reshape((self.time_steps, new_shape))(x)

        # Bidirectional LSTM layers to capture dependencies in both directions
        for _ in range(1):
            x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.1))(x)

        # Multi-Head Attention Layer with fixed number of heads and key dimensions to prevent excessive computational cost
        attention_layer = MultiHeadAttention(num_heads=self.num_heads + len(self.block_configs), key_dim=self.key_dim * 2)
        x = attention_layer(x, x)

        # Flatten and Dropout
        x = Flatten()(x)
        # Fully Connected Layer and Output
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)

        # Output layer with a single neuron (for regression) with L2 regularization
        output = Dense(1, kernel_regularizer=l2(0.001))(x)

        model = Model(inputs=input_layer, outputs=output)

        # Compile the model
        optimizer = AdamW(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

        return model






