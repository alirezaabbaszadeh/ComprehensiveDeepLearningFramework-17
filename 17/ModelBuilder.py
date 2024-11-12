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
    MultiHeadAttention,
    Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
import tensorflow as tf

# Custom Layer: Mix of Experts
class MixOfExperts(tf.keras.layers.Layer):
    def __init__(self, num_experts=10, units=256):  # Increase number of experts and units for maximum capacity
        super(MixOfExperts, self).__init__()
        # Create multiple expert layers, each being a Dense layer
        self.num_experts = num_experts
        self.experts = [Dense(units, activation='relu') for _ in range(num_experts)]
        # Gate layer to determine the weight of each expert
        self.gate = Dense(num_experts, activation='softmax')

    def call(self, x):
        # Compute gate values (weights for each expert)
        gate_values = self.gate(x)
        # Compute outputs from all experts
        expert_outputs = [expert(x) for expert in self.experts]
        # Stack expert outputs along a new dimension
        expert_outputs = tf.stack(expert_outputs, axis=-1)
        # Apply gating to the expert outputs
        gated_output = tf.reduce_sum(expert_outputs * tf.expand_dims(gate_values, -2), axis=-1)
        return gated_output

# Configuration class for Convolutional Layer Blocks
class ConvLayerConfig:
    def __init__(self, block_configs):
        # Store the block configurations
        self.block_configs = block_configs

# Model Builder class to create the entire model
class ModelBuilder:
    def __init__(self, time_steps, num_features, block_configs, num_heads=8, key_dim=256):  # Increase num_heads and key_dim for maximum capacity
        # Initialize model parameters
        self.time_steps = time_steps  # Number of time steps in the input
        self.num_features = num_features  # Number of features per time step
        self.block_configs = block_configs  # Configuration for each convolutional block
        self.num_heads = num_heads  # Number of heads for multi-head attention
        self.key_dim = key_dim  # Dimension of the keys in multi-head attention

    # Residual block with convolution, attention, and shortcut connections
    def residual_block(self, x, filters, kernel_size, pool_size, block_num):
        # Save the input for the shortcut connection
        shortcut = x
        # First convolutional layer
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', name=f'conv1_{block_num}')(x)
        x = LeakyReLU(name=f'leaky_relu1_{block_num}')(x)  # Apply activation function
        x = BatchNormalization(name=f'bn1_{block_num}')(x)  # Normalize to improve training stability
        
        # Multi-Head Attention layer after the first convolution
        attention_layer_1 = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, name=f'attention1_{block_num}')
        x = attention_layer_1(x, x)  # Self-attention to capture dependencies in the features
        
        # Second convolutional layer
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', name=f'conv2_{block_num}')(x)
        x = BatchNormalization(name=f'bn2_{block_num}')(x)  # Normalize again after convolution
        
        # Multi-Head Attention layer after the second convolution
        attention_layer_2 = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, name=f'attention2_{block_num}')
        x = attention_layer_2(x, x)  # Another self-attention layer to capture complex dependencies
        
        # Transforming shortcut if the number of filters does not match
        transformed_shortcut = shortcut
        if shortcut.shape[-1] != filters:
            # Adjust the shortcut to match the shape of the main path
            transformed_shortcut = Conv1D(filters=filters, kernel_size=1, padding='same', name=f'shortcut_conv_{block_num}')(shortcut)
            transformed_shortcut = BatchNormalization(name=f'shortcut_bn_{block_num}')(transformed_shortcut)
              
        # Adding the shortcut to the main path (Residual Connection)
        x = Add(name=f'add_{block_num}')([transformed_shortcut, x])
        x = LeakyReLU(name=f'leaky_relu2_{block_num}')(x)  # Activation after addition
        
        # Optional MaxPooling layer to reduce dimensionality
        if pool_size is not None:
            x = MaxPooling1D(pool_size=pool_size, padding='same', name=f'max_pool_{block_num}')(x)
        
        return x

    # Function to build the model
    def build_model(self):
        # Define the input layer with explicit shape
        input_layer = Input(shape=(self.time_steps, self.num_features))
        
        # Convolutional layers to extract local patterns from the input
        x = input_layer
        for i, block in enumerate(self.block_configs):
            # Extract block configuration details
            filters = block.get('filters', 256)  # Increase number of filters for maximum feature extraction
            kernel_size = block.get('kernel_size', 7)  # Use larger kernel size for capturing more context
            pool_size = block.get('pool_size', 3)  # Use pooling to reduce dimensionality effectively
            # Apply residual block
            x = self.residual_block(x, filters, kernel_size, pool_size, i)
        
        # Bidirectional LSTM layers to capture temporal dependencies in both directions
        for _ in range(2):  # Increase number of LSTM layers for deeper temporal representation
            x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.5, recurrent_dropout=0.3))(x)  # Increase units and dropout for maximum capacity

        # Multi-Head Attention layer to focus on different parts of the sequence
        attention_layer = MultiHeadAttention(num_heads=self.num_heads + len(self.block_configs), key_dim=self.key_dim * 2)
        x = attention_layer(x, x)  # Self-attention layer to further capture temporal dependencies

        # Mix of Experts layer to combine different expert predictions
        mix_of_experts = MixOfExperts(num_experts=10, units=256)  # Use more experts and units for high capacity
        x = mix_of_experts(x)  # Apply Mix of Experts to the output
        x = Dropout(0.5)(x)  # Increase dropout rate for regularization

        # Flatten the output before the fully connected layers
        x = Flatten()(x)

        # Fully connected layer to learn final representations
        x = Dense(256, activation='relu')(x)  # Increase units for better learning capacity
        x = Dropout(0.5)(x)  # Increase dropout for regularization
        # Output layer with L2 regularization
        output = Dense(1, kernel_regularizer=l2(0.01))(x)  # Increase regularization to prevent overfitting

        # Create the model
        model = Model(inputs=input_layer, outputs=output)

        # Compile the model with AdamW optimizer
        optimizer = AdamW(learning_rate=0.0001)  # Reduce learning rate for more stable convergence
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])  # Compile with Mean Squared Error loss and Mean Absolute Error metric

        return model

# Hyperparameter Guide:
# - time_steps: The number of time steps in each input sequence. This value depends on the data you are using. Typical values are between 50 to 500.
#   - تعداد گام‌های زمانی در هر توالی ورودی. این مقدار بستگی به داده‌های مورد استفاده دارد. مقادیر معمول بین 50 تا 500 است.

# - num_features: The number of features in each time step. This value should match the number of features extracted from the dataset. Typical values range from 10 to 100.
#   - تعداد ویژگی‌ها در هر گام زمانی. این مقدار باید با تعداد ویژگی‌های استخراج شده از دیتاست مطابقت داشته باشد. مقادیر معمول بین 10 تا 100 است.

# - block_configs: A list of dictionaries, each specifying the configuration for a convolutional block.
#   - filters: Number of filters in the convolutional layers. Higher values allow more complex features to be learned. Typical values range from 32 to 256.
#     - تعداد فیلترها در لایه‌های کانولوشنی. مقادیر بیشتر امکان یادگیری ویژگی‌های پیچیده‌تر را فراهم می‌کنند. مقادیر معمول بین 32 تا 256 است.

#   - kernel_size: The size of the convolutional kernel. Larger kernels capture more context but can be more computationally expensive. Typical values are 3, 5, or 7.
#     - اندازه کرنل کانولوشنی. کرنل‌های بزرگ‌تر امکان دریافت اطلاعات بیشتر از بافت را فراهم می‌کنند، اما محاسبات بیشتری نیاز دارند. مقادیر معمول 3، 5 یا 7 هستند.

#   - pool_size: Size of the pooling operation to downsample the data. If None, no pooling is applied. Common values are 2 or 3.
#     - اندازه عملیات پولاوری برای کاهش اندازه داده‌ها. اگر None باشد، پولاوری اعمال نمی‌شود. مقادیر رایج 2 یا 3 هستند.

# - num_heads: Number of attention heads in the Multi-Head Attention layers. More heads allow the model to learn different types of relationships. Typical values are between 2 to 8.
#   - تعداد هدهای توجه در لایه‌های Multi-Head Attention. هدهای بیشتر به مدل امکان یادگیری انواع مختلف روابط را می‌دهند. مقادیر معمول بین 2 تا 8 است.

# - key_dim: The dimensionality of the key vectors in the attention mechanism. Larger values increase the model's ability to capture fine-grained details. Common values are 64, 128, or 256.
#   - ابعاد بردارهای کلید در مکانیزم توجه. مقادیر بزرگ‌تر توانایی مدل در دریافت جزئیات دقیق را افزایش می‌دهند. مقادیر رایج 64، 128 یا 256 هستند.

# - LSTM units: The number of units in the LSTM layers. More units can help capture more complex temporal patterns. Typical values are 64, 128, or 256.
#   - تعداد واحدها در لایه‌های LSTM. واحدهای بیشتر می‌توانند الگوهای زمانی پیچیده‌تری را به‌دست آورند. مقادیر معمول 64، 128 یا 256 هستند.

# - Dropout rates: The dropout rate used in Dropout layers to reduce overfitting by randomly setting a fraction of input units to zero during training. Typical values range from 0.1 to 0.5.
#   - نرخ دراپ‌اوت که در لایه‌های Dropout برای کاهش بیش‌برازش با تنظیم تصادفی بخشی از واحدهای ورودی به صفر در طول آموزش استفاده می‌شود. مقادیر معمول بین 0.1 تا 0.5 هستند.

# - Learning rate: The learning rate for the AdamW optimizer. Lower values lead to more stable convergence, but training may be slower. Common values are 0.001, 0.0005, or 0.0001.
#   - نرخ یادگیری برای بهینه‌ساز AdamW. مقادیر پایین‌تر به همگرایی پایدارتر منجر می‌شوند، اما ممکن است آموزش کندتر شود. مقادیر رایج 0.001، 0.0005 یا 0.0001 هستند.

# - Regularization (L2): The L2 regularization parameter to prevent overfitting. Typical values are between 0.0001 to 0.01.
#   - پارامتر منظم‌سازی L2 برای جلوگیری از بیش‌برازش. مقادیر معمول بین 0.0001 تا 0.01 هستند.

# - Number of experts: The number of experts in the MixOfExperts layer. More experts provide greater model capacity but increase computational cost. Typical values are 3 to 10.
#   - تعداد کارشناسان در لایه MixOfExperts. کارشناسان بیشتر ظرفیت مدل را افزایش می‌دهند اما هزینه محاسباتی را نیز بالا می‌برند. مقادیر معمول بین 3 تا 10 هستند.
