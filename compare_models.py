import joblib
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score

# -----------------------------
# 1. بارگذاری و آماده‌سازی داده
# -----------------------------
data = load_breast_cancer()
X = data.data
y = data.target
target_names = data.target_names  # ['malignant', 'benign']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 2. تعریف مدل‌ها
# -----------------------------
models = {
    "GaussianNB": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
}

# -----------------------------
# 3. مقایسه با Cross-Validation
# -----------------------------
results = []

print("=" * 60)
print("مقایسه مدل‌ها (Cross-Validation روی داده آموزش)")
print("=" * 60)

for name, model in models.items():
    # دقت با 5-fold CV
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
    
    # آموزش روی کل train و ارزیابی روی test
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    # recall برای کلاس بدخیم (0) مهمه — چون نباید سرطان رو از دست بدیم
    recall_malignant = recall_score(y_test, y_pred, pos_label=0)
    
    results.append({
        "Model": name,
        "CV Accuracy (mean)": round(cv_scores.mean() * 100, 2),
        "CV Std": round(cv_scores.std() * 100, 2),
        "Test Accuracy": round(acc * 100, 2),
        "Recall (Malignant)": round(recall_malignant * 100, 2),
    })
    
    print(f"\n{name}")
    print(f"  CV Accuracy : {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f})")
    print(f"  Test Accuracy: {acc*100:.2f}%")
    print(f"  Recall بدخیم : {recall_malignant*100:.2f}%")

# -----------------------------
# 4. جدول نتایج
# -----------------------------
df_results = pd.DataFrame(results)
df_results = df_results.sort_values("Test Accuracy", ascending=False)

print("\n" + "=" * 60)
print("جدول نهایی (مرتب‌شده بر اساس دقت تست)")
print("=" * 60)
print(df_results.to_string(index=False))

# -----------------------------
# 5. ذخیره بهترین مدل
# -----------------------------
best_name = df_results.iloc[0]["Model"]
best_model = models[best_name]
best_model.fit(X_train_scaled, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, os.path.join("models", "best_model.joblib"))
joblib.dump(scaler, os.path.join("models", "scaler.joblib"))

print(f"\nبهترین مدل: {best_name}")
print("ذخیره شد → models/best_model.joblib")
print("Scaler ذخیره شد → models/scaler.joblib")

# گزارش کامل بهترین مدل
y_pred_best = best_model.predict(X_test_scaled)
print("\nگزارش کامل بهترین مدل:")
print(classification_report(y_test, y_pred_best, target_names=target_names))