import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# 1. Kaydettiğimiz Temiz Verileri Yükleme
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze() # Squeeze ile Series'e çeviriyoruz
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print("Veriler başarıyla yüklendi! Lojistik Regresyon modeli eğitiliyor...\n")

# 2. Modeli Tanımlama ve Eğitme
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)

# 3. Tahminlerin Yapılması
y_pred = log_model.predict(X_test)
y_pred_proba = log_model.predict_proba(X_test)[:, 1] # ROC Eğrisi için olasılıklar

# 4. Metrik Raporunun Yazdırılması
print("--- LOJİSTİK REGRESYON PERFORMANS RAPORU ---")
print(classification_report(y_test, y_pred))


# ---------------------------------------------------------
# 5. GÖRSELLEŞTİRMELER
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Görsel 1: Karmaşıklık Matrisi (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Karmaşıklık Matrisi (Confusion Matrix)')
axes[0].set_xlabel('Tahmin Edilen')
axes[0].set_ylabel('Gerçek Değer')


# Görsel 2: ROC Eğrisi
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_title('ROC Eğrisi')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc="lower right")

# Görsel 3: Özellik Katsayıları (En Etkili 10 Değişken)
# Modelin kararlarını neye göre verdiğini anlıyoruz
coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': log_model.coef_[0]
})
# Etki büyüklüğüne göre (mutlak değer) sırala
coefficients['Abs_Importance'] = coefficients['Importance'].abs()
top_features = coefficients.sort_values(by='Abs_Importance', ascending=False).head(10)

sns.barplot(x='Importance', y='Feature', data=top_features, ax=axes[2], palette='viridis')
axes[2].set_title('Müşteri Kaybına En Çok Etki Eden 10 Özellik')
axes[2].set_xlabel('Katsayı (Etki Yönü ve Büyüklüğü)')
axes[2].set_ylabel('Özellik (Feature)')

plt.tight_layout()

#grafiğin kaydedilmesi

plt.tight_layout()
plt.savefig("reports/logreg_results.png", dpi=300) # Kaydetme satırı
plt.show()



