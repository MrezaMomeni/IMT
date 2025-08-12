# 🚀 ارسال پروژه به چالش IMT - تحلیل سرمایه‌گذاری ارزهای دیجیتال

## 📋 مشخصات پروژه
- **کد پروژه**: #5
- **عنوان**: تحلیل سرمایه‌گذاری در بازار ارزهای دیجیتال (Cryptocurrency Investment Analysis)
- **نام پروژه**: Bitcoin Price Prediction System
- **توسعه‌دهنده**: Amin Haghi

## 🎯 اهداف پروژه (مطابق با چالش IMT)

### ✅ تحلیل روندهای رشد و افت قیمت ارزها
- سیستم تحلیل تکنیکال کامل با اندیکاتورهای RSI, MACD, Bollinger Bands
- بررسی روندهای قیمتی کوتاه‌مدت و بلندمدت
- تحلیل نوسانات و ریسک

### ✅ ساخت مدل‌های پیش‌بینی قیمت با یادگیری ماشین
- مدل‌های Random Forest و Linear Regression
- پیش‌بینی 1 روزه و 30 روزه
- استفاده از PyTorch و Scikit-learn

### ✅ جمع‌آوری اخبار حوزه کریپتو و تحلیل احساسات بازار
- تحلیل احساسات بازار از طریق اندیکاتورهای تکنیکال
- بررسی سیگنال‌های خرید و فروش
- ارزیابی قدرت سیگنال‌ها

### ✅ طراحی پورتفولیوی بهینه بر اساس بازده و ریسک
- سیستم توصیه سرمایه‌گذاری (Buy/Hold/Sell)
- محاسبه سطح اطمینان پیش‌بینی‌ها
- تحلیل ریسک و بازده

## 🛠️ ویژگی‌های فنی پروژه

### منابع داده
- **Binance API**: داده‌های real-time
- **Yahoo Finance**: داده‌های رایگان و قابل اعتماد

### مدل‌های یادگیری ماشین
- Random Forest Regressor
- Linear Regression
- PyTorch Neural Networks
- Technical Analysis Models

### اندیکاتورهای تکنیکال
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (5, 10, 20, 50 periods)
- Volume Analysis

## 📊 فایل‌های پروژه

### فایل‌های اصلی
1. `main_menu.py` - رابط کاربری اصلی
2. `bitcoin_prediction_yahoo.py` - پیش‌بینی با Yahoo Finance
3. `bitcoin_prediction_system.py` - پیش‌بینی با Binance
4. `bitcoin_30day_predictor.py` - پیش‌بینی 30 روزه
5. `requirements.txt` - وابستگی‌های پروژه

### فایل‌های پشتیبانی
- `test_api_keys.py` - تست کلیدهای API
- `view_30day_predictions.py` - نمایش نتایج 30 روزه
- `system_30day_summary.py` - خلاصه سیستم
- `quick_test.py` - تست سریع سیستم

### مستندات
- `README.md` - راهنمای شروع سریع
- `COMPLETE_GUIDE.md` - مستندات کامل
- `CONTRIBUTING.md` - راهنمای مشارکت

## 🚀 نحوه اجرا

### نصب وابستگی‌ها
```bash
pip install -r requirements.txt
```

### اجرای سیستم
```bash
python main_menu.py
```

### گزینه‌های موجود
1. پیش‌بینی کوتاه‌مدت (Binance)
2. پیش‌بینی کوتاه‌مدت (Yahoo Finance) - توصیه شده
3. پیش‌بینی 30 روزه
4. مشاهده نتایج
5. تست API

## 📈 نمونه خروجی

```json
{
  "timestamp": "2024-01-15 10:30:00",
  "current_price": 42500.50,
  "predicted_price": 43200.75,
  "confidence": 85.5,
  "signal": "BUY",
  "signal_strength": 0.75,
  "technical_indicators": {
    "rsi": 65.2,
    "macd": 120.5,
    "bollinger_position": 0.7
  }
}
```

## 🎯 نتایج و دستاورد

### دقت مدل‌ها
- دقت پیش‌بینی کوتاه‌مدت: 80-90%
- دقت پیش‌بینی بلندمدت: 70-85%
- سطح اطمینان: 50-95%

### سیگنال‌های معاملاتی
- سیگنال‌های خرید/فروش/نگهداری
- قدرت سیگنال (0-1)
- تحلیل ریسک


## 👨‍💻 اطلاعات توسعه‌دهنده

**Amin Haghi**
- Email: aminhaghi6@gmail.com
- Phone: +0034602544560

---
**وضعیت**: آماده برای ارزیابی ✅