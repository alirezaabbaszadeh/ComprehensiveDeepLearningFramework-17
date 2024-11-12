import tensorflow as tf

# تنظیم استفاده فقط از GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # تعیین استفاده از GPU
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU")
    except RuntimeError as e:
        print(e)
else:
    print("Using CPU")
