import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# بارگذاری داده
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['diagnosis'] = df['target'].map({0: 'malignant', 1: 'benign'})

print("شکل داده:", df.shape)
print("\nتعداد هر کلاس:")
print(df['diagnosis'].value_counts())

# -----------------------------
# 1. توزیع کلاس‌ها
# -----------------------------
counts = df['diagnosis'].value_counts()
colors = ['#e74c3c', '#2ecc71']

plt.figure(figsize=(7, 5))
bars = plt.bar(counts.index, counts.values, color=colors)
plt.title('توزیع کلاس‌ها (بدخیم vs خوش‌خیم)')
plt.xlabel('تشخیص')
plt.ylabel('تعداد')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.show()

# -----------------------------
# 2. هیستوگرام چند ویژگی مهم
# -----------------------------
important_features = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness',
    'mean compactness'
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()

for i, feature in enumerate(important_features):
    ax = axes[i]
    malignant = df[df['diagnosis'] == 'malignant'][feature]
    benign = df[df['diagnosis'] == 'benign'][feature]

    ax.hist(malignant, bins=25, alpha=0.6, color='#e74c3c', label='malignant')
    ax.hist(benign, bins=25, alpha=0.6, color='#2ecc71', label='benign')
    ax.set_title(feature)
    ax.legend()

plt.suptitle('هیستوگرام ویژگی‌های مهم (مقایسه دو کلاس)', fontsize=16)
plt.tight_layout()
plt.savefig('histograms.png', dpi=150)
plt.show()

# -----------------------------
# 3. Heatmap همبستگی (با matplotlib)
# -----------------------------
corr = df[data.feature_names].corr().values

plt.figure(figsize=(12, 10))
im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.title('نقشه حرارتی همبستگی ویژگی‌ها', fontsize=14)

# چون ۳۰ تا ویژگی داریم، اسم‌ها رو نشون نمی‌دیم تا شلوغ نشه
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()

print("\nنمودارها ذخیره شدند:")
print("- class_distribution.png")
print("- histograms.png")
print("- correlation_heatmap.png")