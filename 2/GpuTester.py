import tensorflow as tf

# بررسی دستگاه‌های در دسترس
print("Available devices:", tf.config.list_physical_devices())

# بررسی اینکه آیا GPU در حال استفاده است یا خیر
if tf.config.list_physical_devices('GPU'):
    print("GPU is available and being used.")
else:
    print("GPU is not being used, only CPU is available.")
