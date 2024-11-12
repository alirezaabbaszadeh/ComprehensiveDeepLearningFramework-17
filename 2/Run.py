# run.py
from MainClass import TimeSeriesModel
import os


base_dir = os.path.dirname(os.path.abspath(__file__))   

# تنظیم پارامترهای مدل
time_series_model = TimeSeriesModel(
    file_path=os.path.join(base_dir, "csv/EURUSD_Candlestick_1_Hour_BID_01.01.2015-28.09.2024.csv"),
    base_dir=base_dir,
    epochs=3, 
    batch_size=128
)

# اجرای کل فرآیند آموزش و ارزیابی
time_series_model.run()                








