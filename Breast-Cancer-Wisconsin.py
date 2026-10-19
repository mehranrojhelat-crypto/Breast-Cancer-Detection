import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# بارگذاری دیتاست
data = load_breast_cancer()
X = data.data
y = data.target
target_names = data.target_names  # ['malignant', 'benign']

print("تعداد نمونه‌ها:", X.shape[0])
print("تعداد ویژگی‌ها:", X.shape[1])
print("کلاس‌ها:", list(target_names))
print("-" * 50)

# تقسیم داده
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# نرمال‌سازی
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# آموزش مدل
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# پیش‌بینی
y_pred = model.predict(X_test_scaled)

# نتایج
accuracy = accuracy_score(y_test, y_pred)
print(f"دقت مدل: {accuracy * 100:.2f}%")
print("\nگزارش کامل:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("ماتریس درهم‌ریختگی:")
print(confusion_matrix(y_test, y_pred))


# تابع پیش‌بینی
def predict_breast_cancer(features):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    result = target_names[prediction]
    print(f"\nنتیجه تشخیص: {result}")
    print(f"احتمال بدخیم (malignant): {probability[0] * 100:.2f}%")
    print(f"احتمال خوش‌خیم (benign): {probability[1] * 100:.2f}%")
    return result


# تست با یک نمونه واقعی
print("\n" + "=" * 50)
print("تست روی یک نمونه واقعی از دیتاست:")
predict_breast_cancer(X_test[0])

print("\nبرنامه با موفقیت اجرا شد.")