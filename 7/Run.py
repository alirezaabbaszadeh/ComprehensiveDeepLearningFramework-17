# run.py
from MainClass import TimeSeriesModel
import os

# Example configuration for convolutional blocks
block_configs = [
    {'filters': 16, 'kernel_size': (2, 2, 1), 'pool_size': 8},
    # {'filters': 128, 'kernel_size': (3, 3, 3), 'pool_size': 2},
    # {'filters': 256, 'pool_size': 2},
    # {'filters': 512},
    {'filters': 16, 'kernel_size': (2, 2, 1), 'pool_size': None}  # بدون MaxPooling در آخرین بلوک
]

base_dir = os.path.dirname(os.path.abspath(__file__))   

# تنظیم پارامترهای مدل
time_series_model = TimeSeriesModel(
    file_path=os.path.join(base_dir, "csv/EURUSD_Candlestick_1_Hour_BID_01.01.2015-28.09.2024.csv"),
    base_dir=base_dir,
    epochs=3, 
    batch_size=256,
    block_configs=block_configs  # ارسال تنظیمات بلوک‌ها
)

# اجرای کل فرآیند آموزش و ارزیابی
time_series_model.run()                








