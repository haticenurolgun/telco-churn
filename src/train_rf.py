import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# 1. Temiz Verileri Yükleme
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print("Veriler yüklendi! Random Forest modeli eğitiliyor...\n")

# 2. Modeli Tanımlama ve Eğitme (class_weight='balanced' ile dengesizliği çözüyoruz)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# 3. Tahminlerin Yapılması
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# 4. Rapor
print("--- RANDOM FOREST PERFORMANS RAPORU ---")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 5. GÖRSELLEŞTİRMELER
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Görsel 1: Karmaşıklık Matrisi
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0])
axes[0].set_title('RF Karmaşıklık Matrisi')
axes[0].set_xlabel('Tahmin Edilen')
axes[0].set_ylabel('Gerçek Değer')

# Görsel 2: ROC Eğrisi
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color='green', lw=2, label=f'AUC = {roc_auc:.2f}')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_title('RF ROC Eğrisi')
axes[1].legend(loc="lower right")

# Görsel 3: Özellik Önem Derecesi (Feature Importance)
importances = pd.DataFrame({
    'Feature': X_train.columns, 
    'Importance': rf_model.feature_importances_
})
top_features = importances.sort_values(by='Importance', ascending=False).head(10)

sns.barplot(x='Importance', y='Feature', data=top_features, ax=axes[2], palette='viridis')
axes[2].set_title('RF En Önemli 10 Özellik')

plt.tight_layout()

plt.tight_layout()
plt.savefig("reports/rf_results.png", dpi=300) # Kaydetme satırı
plt.show()
