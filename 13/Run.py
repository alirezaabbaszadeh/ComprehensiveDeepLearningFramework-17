# run.py
from MainClass import TimeSeriesModel
import os

# تعریف تنظیمات برای هر بلوک
# block_configs = [
#     {'filters': 64, 'kernel_size': 5, 'pool_size': 3},  # به طور صریح kernel_size و pool_size مقداردهی شده‌اند
#     {'filters': 128, 'kernel_size': 4, 'pool_size': 2},  # kernel_size و pool_size خاصی تعیین شده‌اند
#     {'filters': 256, 'pool_size': 2},  # فقط pool_size تعیین شده است، مقدار پیش‌فرض kernel_size اعمال می‌شود (3)
#     {'filters': 512},  # نه kernel_size و نه pool_size تعیین نشده‌اند، مقدار پیش‌فرض هر دو اعمال می‌شود
# ]

block_configs = [
    {'filters': 64, 'kernel_size': 5, 'pool_size': 4},
    # {'filters': 128, 'kernel_size': 3, 'pool_size': 2},
    # {'filters': 256, 'kernel_size': 5, 'pool_size': 2},
    # {'filters': 512, 'kernel_size': 5, 'pool_size': 2},
    # {'filters': 1024, 'kernel_size': 3, 'pool_size': 2},
    # {'filters': 1024, 'kernel_size': 3, 'pool_size': 2},
    # {'filters': 1024, 'kernel_size': 3, 'pool_size': 2},
    {'filters': 64, 'kernel_size': 2, 'pool_size': None}  # بدون MaxPooling در آخرین بلوک
]
base_dir = os.path.dirname(os.path.abspath(__file__))   

# تنظیم پارامترهای مدل
time_series_model = TimeSeriesModel(
    file_path=os.path.join(base_dir, "csv/EURUSD_Candlestick_1_Hour_BID_01.01.2015-28.09.2024.csv"),
    base_dir=base_dir,
    epochs=3, 
    batch_size=64,
    block_configs=block_configs  # ارسال تنظیمات بلوک‌ها
)

# اجرای کل فرآیند آموزش و ارزیابی
time_series_model.run()                








