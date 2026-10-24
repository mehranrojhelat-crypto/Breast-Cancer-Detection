markdown
# Breast Cancer Detection

تشخیص خوش‌خیم / بدخیم بودن تومور سینه با یادگیری ماشین  
دیتاست: **Breast Cancer Wisconsin** (sklearn)

## ویژگی‌ها
- تحلیل و مصورسازی داده (`eda.py`)
- مقایسه مدل‌ها: GaussianNB, KNN, RandomForest, GradientBoosting, AdaBoost
- Tuning با GridSearchCV
- ذخیره مدل نهایی با joblib
- رابط کاربری با Streamlit

## ساختار پروژه
```
Breast-Cancer-Detection/
├── models/
│   ├── best_model.joblib
│   └── scaler.joblib
├── eda.py
├── train_and_save.py
├── compare_models.py
├── tune_model.py
├── app.py
└── README.md
```

## نصب
```bash
pip install numpy pandas scikit-learn matplotlib streamlit joblib
```

## اجرا

### ۱) مصورسازی داده
```bash
python eda.py
```

### ۲) آموزش و Tuning
```bash
python tune_model.py
```

### ۳) رابط کاربری
```bash
streamlit run app.py
```

## نکته مهم
این پروژه **آموزشی** است و نباید برای تشخیص پزشکی واقعی استفاده شود.
```