import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# โหลดข้อมูลจากไฟล์ Excel
df = pd.read_excel("CJ_Final.xlsx", sheet_name="Sales Data")

# เลือกเฉพาะวันที่และยอดขาย
df = df[['Date', 'Total Sales']].dropna()
df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'Date': 'ds', 'Total Sales': 'y'})

# สร้างโมเดล ARIMA
model = ARIMA(df['y'], order=(5,1,0))
model_fit = model.fit()

# ทำนายยอดขายล่วงหน้า 90 วัน
forecast = model_fit.forecast(steps=90)

# สร้าง DataFrame สำหรับผลลัพธ์
future_dates = pd.date_range(start=df['ds'].max(), periods=90, freq='D')
forecast_df = pd.DataFrame({'ds': future_dates, 'Predicted Sales': forecast})

# แสดงกราฟ
plt.figure(figsize=(10, 5))
plt.plot(df['ds'], df['y'], label="Actual Sales", color='blue')
plt.plot(forecast_df['ds'], forecast_df['Predicted Sales'], label="Predicted Sales", linestyle="dashed", color='red')
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.title("Sales Forecasting (ARIMA Model)")
plt.legend()
plt.grid()
plt.show()

# บันทึกผลลัพธ์
forecast_df.to_csv("Sales_Forecast.csv", index=False)
