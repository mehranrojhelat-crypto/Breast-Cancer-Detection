import joblib
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score

# -----------------------------
# 1. داده
# -----------------------------
data = load_breast_cancer()
X, y = data.data, data.target
target_names = data.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 2. مدل‌ها + محدوده پارامترها
# -----------------------------
param_grids = {
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
        },
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
        },
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 150],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        },
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.5, 1.0],
        },
    },
}

# -----------------------------
# 3. GridSearch برای هر مدل
# -----------------------------
results = []
best_overall_score = -1
best_overall_model = None
best_overall_name = None

print("=" * 60)
print("شروع Tuning (ممکن است چند دقیقه طول بکشد)")
print("=" * 60)

for name, config in param_grids.items():
    print(f"\nدر حال Tuning: {name} ...")

    grid = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train_scaled, y_train)

    y_pred = grid.best_estimator_.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    recall_mal = recall_score(y_test, y_pred, pos_label=0)

    results.append({
        "Model": name,
        "Best CV Accuracy": round(grid.best_score_ * 100, 2),
        "Test Accuracy": round(test_acc * 100, 2),
        "Recall (Malignant)": round(recall_mal * 100, 2),
        "Best Params": grid.best_params_,
    })

    print(f"  بهترین CV Accuracy: {grid.best_score_*100:.2f}%")
    print(f"  Test Accuracy     : {test_acc*100:.2f}%")
    print(f"  Recall بدخیم      : {recall_mal*100:.2f}%")
    print(f"  بهترین پارامترها  : {grid.best_params_}")

    if grid.best_score_ > best_overall_score:
        best_overall_score = grid.best_score_
        best_overall_model = grid.best_estimator_
        best_overall_name = name

# -----------------------------
# 4. جدول نتایج
# -----------------------------
df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
print("\n" + "=" * 60)
print("نتایج نهایی Tuning")
print("=" * 60)
print(df[["Model", "Best CV Accuracy", "Test Accuracy", "Recall (Malignant)"]].to_string(index=False))

# -----------------------------
# 5. ذخیره بهترین مدل
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(best_overall_model, os.path.join("models", "best_model.joblib"))
joblib.dump(scaler, os.path.join("models", "scaler.joblib"))

print(f"\nبرنده نهایی بعد از Tuning: {best_overall_name}")
print("ذخیره شد → models/best_model.joblib")

y_pred_final = best_overall_model.predict(X_test_scaled)
print("\nگزارش کامل مدل نهایی:")
print(classification_report(y_test, y_pred_final, target_names=target_names))