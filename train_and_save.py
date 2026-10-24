import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. بارگذاری داده
data = load_breast_cancer()
X = data.data
y = data.target
target_names = data.target_names

# 2. تقسیم داده
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. نرمال‌سازی
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. آموزش مدل
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. ارزیابی سریع
y_pred = model.predict(X_test_scaled)
print(f"دقت مدل: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=target_names))

# 6. ذخیره مدل و scaler
joblib.dump(model, "breast_cancer_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("\nمدل ذخیره شد → breast_cancer_model.joblib")
print("Scaler ذخیره شد → scaler.joblib")