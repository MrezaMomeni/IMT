
# `README_FA.md` (نسخه فارسی)

# 🩺 تشخیص سنگ کلیه با یادگیری عمیق

> **Notebook در Kaggle:** Kidney Stone Classification — Task 1  
> لینک: https://www.kaggle.com/code/sepehrkh/kidney-stone-classification-task-1-tensorfollow

---

## 📌 معرفی پروژه

این ریپازیتوری شامل کدها، نتایج و خروجی‌های مربوط به **تشخیص سنگ کلیه** (طبقه‌بندی دودویی: `Normal` در برابر `Stone`) است.  
هدف، آموزش و مقایسهٔ چند شبکهٔ عصبی کانولوشنی و تهیهٔ گزارش ارزیابی دقیق (Accuracy, Precision, Recall, F1, ROC-AUC, LogLoss) و ارائهٔ آثار قابل استفاده (مدل‌ها، CSV پیش‌بینی‌ها، تاریخچهٔ آموزش) است.

این کار بخشی از چالش‌های برنامهٔ کارآموزی IMT است و به‌صورت قابل اجرا در Kaggle طراحی شده است.

---

## 🎯 هدف پروژه

* طبقه‌بندی دودویی تصاویر پزشکی برای تشخیص سنگ کلیه.  
* مقایسهٔ معماری‌های مختلف (از شبکهٔ ساده تا مدل‌های transfer-learning).  
* انجام ارزیابی جامع، استفاده از روش‌های ensemble (hard & soft voting)، و ذخیرهٔ خروجی‌های قابل استفاده.

فاز بعدی (Phase 2) شامل تشخیص شیء (Object Detection) با YOLO برای محلی‌سازی مناطق سنگ می‌شود؛ این ریپازیتوری مربوط به فاز ۱ (classification) است.

---

## 🗂 ساختار پیشنهادی ریپازیتوری

KidneyStone-Classification/
├── notebooks/ # نوت‌بوک‌های Kaggle / Jupyter (کاوش، آموزش، ارزیابی)
├── src/ # کد ماژولار: data, models, train, utils
├── assets/ # تصاویر نمونه و نمودارها
├── results/ # CSV پیش‌بینی‌ها، .npz تاریخچه‌ها، نمودارها
├── models/ # مدل‌های ذخیره‌شده (.keras) — معمولاً به GitHub push نمی‌شوند
├── configs/ # فایل‌های تنظیمات (yaml)
├── logs/ # لاگ‌های اجرا
├── README.md # این فایل (انگلیسی)
├── README_FA.md # این فایل (فارسی)
├── requirements.txt # وابستگی‌ها
└── .gitignore # فایل‌هایی که نباید پوش شوند


---

## 📊 مجموعه داده

* **منبع:** دیتاست Kidney Stone (Kaggle — لینک در نوت‌بوک).  
* ساختار معمول: `df_train`, `df_val`, `df_test` (ستون `image_path` و `label` برای برچسب‌دارها).  
* `testDataset` — بدون برچسب، برای پیش‌بینی نهایی.  
* `testDataset_with_labels` — برای ارزیابی، با `shuffle=False` ساخته شده تا ترتیب نمونه‌ها با CSV هم‌راستا بماند.

---

## 🧪 مدل‌های آزمایش‌شده

* `simple_cnn_model` — شبکهٔ ساده‌ و سریع (baseline)  
* `AlexNet_model` — معماری کلاسیک  
* `VGGNet_model` — معماری عمیق کانولوشنی (نتایج عالی)  
* `ResNet34_model` — شبکهٔ residual (در این دیتاست ضعیف‌تر بود)  
* `EfficientNetB0/B1/B2` — تست شد، اما در اجراهای ما به‌صورت تک‌کلاسه (`Stone`) رفتار کردند (مشخصات در بخش نکات)

---

## ⚙ پیکربندی آموزش (استفاده‌شده)

* **بهینه‌ساز:** Adam  
* **نرخ یادگیری:** `1e-4`  
* **تابع زیان:** `categorical_crossentropy`  
* **متریک:** `accuracy` (سایر متریک‌ها در ارزیابی محاسبه شدند)  
* **Callbacks:** `ReduceLROnPlateau`, `EarlyStopping(restore_best_weights=True)`, `ModelCheckpoint(save_best_only=True)`  
* **پایپ‌لاین داده:** تبدیل `ImageDataGenerator` → `tf.data` + `prefetch(AUTOTUNE)`  
* **ذخیره خروجی:** هر مدل CSV پیش‌بینی (برچسب سخت + احتمال‌های softmax)، تاریخچهٔ `.npz` و `meta.json`.

---

## 📈 نتایج نهایی (خلاصه)

متریک‌ها بر اساس `testDataset_with_labels` و با استفاده از CSVهای ذخیره‌شده محاسبه شده‌اند.

| Model                     | Accuracy | Precision | Recall  | F1      | ROC_AUC | LogLoss |
|---------------------------|---------:|----------:|--------:|--------:|--------:|--------:|
| **AlexNet_model**         | 0.9868   | 0.9802    | 0.9955  | 0.9878  | 0.9996  | 0.0471  |
| **VGGNet_model**          | 0.9844   | 0.9888    | 0.9821  | 0.9854  | 0.9988  | 0.0583  |
| **simple_cnn_model**      | 0.9663   | 0.9412    | 1.0000  | 0.9697  | 0.9999  | 0.1213  |
| **ResNet34_model**        | 0.7236   | 0.7433    | 0.7433  | 0.7433  | 0.8103  | 0.5767  |
| **Ensemble (Hard Voting)**| 0.9808   | 0.9800    | 0.9844  | 0.9822  |  —     |  —      |
| **Ensemble (Soft Voting)**| 0.9880   | 0.9782    | 1.0000  | 0.9890  | 0.9999  | 0.1579  |

**نکات کلیدی**
* `AlexNet` و `VGGNet` عملکرد بسیار خوب و پایدار داشتند.  
* شبکهٔ ساده (`simple_cnn_model`) با وجود سادگی، نتایج قابل‌توجهی ارائه کرد (Recall = 1.0 در این اجرا).  
* `ResNet34` عملکرد پایینی داشت — نیاز به بررسی بیشتر (تنظیمات، overfitting، داده).  
* **Ensemble (Soft Voting)** بهترین F1 را نشان داد و بیشترین پایداری را داشت.  
* **مسئلهٔ EfficientNet:** در اجراهای ما این مدل‌ها تقریباً همیشه کلاس `Stone` را پیش‌بینی کردند — نشان‌دهندهٔ نیاز به تنظیمات خاص‌تر برای این معماری‌ها.

---

## ⚠ نکات مهم و مشکلات شناخته‌شده

* **EfficientNet collapse:** اگر بخواهید از EfficientNet استفاده کنید، پیشنهاد می‌شود:
  * تقویت داده (stronger augmentation)،
  * تنظیم نرخ یادگیری (warmup/decay)،
  * regularization (weight decay / dropout) و
  * بررسی preprocessing مورد نیاز مدل‌های EfficientNet.
* **حجم داده و GitHub:** داده‌ها و مدل‌های سنگین را به GitHub push نکنید؛ بجای آن از Kaggle یا ذخیرهٔ خارجی استفاده کنید.
* **ارزیابی reproducible:** هنگام محاسبهٔ معیارها از `shuffle=False` برای generatorهای ارزیابی استفاده کنید و `steps` را مشخص نمایید.

---

## 🔁 اجرا و بازتولید (Reproduce)

1. کلون کردن:
```bash
git clone https://github.com/sepehr-khb/KidneyStone-Classification.git
cd KidneyStone-Classification

نصب وابستگی‌ها:

pip install -r requirements.txt


اجرای نوت‌بوک در Kaggle:

باز کردن نوت‌بوک: https://www.kaggle.com/code/sepehrkh/kidney-stone-classification-task-1-tensorfollow

در Kaggle: Run All → Quick Save → سپس Share → Public برای ثبت نسخهٔ اجراپذیر.

اجرای محلی (نمونه):

python src/train.py --model VGGNet_model --config configs/hparams.yaml
python src/evaluate.py --predictions_dir results/

🔮 کارهای بعدی (پیشنهادها)

فاز ۲ — تشخیص شیء (YOLO) برای محلی‌سازی سنگ‌ها (برچسب‌گذاری با CVAT یا Roboflow).

بهبود runهای EfficientNet با تنظیمات و augmentation قوی‌تر.

افزایش اندازهٔ ورودی برای مدل‌های بزرگ‌تر (مثلاً 224×224).

اضافه کردن توضیحات تفسیرپذیری (Grad-CAM) برای اعتماد بالینی.

🧾 مجوز

این پروژه تحت MIT License منتشر می‌شود.

👨‍💻 نویسنده

Sepehr Khalili — GitHub: https://github.com/sepehr-khb

Kaggle Notebook: https://www.kaggle.com/code/sepehrkh/kidney-stone-classification-task-1-tensorfollow

تاریخ: سپتامبر 2025
