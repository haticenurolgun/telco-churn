import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# 1. Temiz Verileri Yükleme
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print("Veriler yüklendi! XGBoost eğitiliyor...\n")

# 2. Dengesiz Veri İçin Ağırlık Hesaplama (Kalanlar / Gidenler oranı)
# Bizim verimizde yaklaşık 3 kat fark var (1053 / 352)
pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# 3. Modeli Tanımlama ve Eğitme
xgb_model = XGBClassifier(
    n_estimators=100, 
    max_depth=4, 
    learning_rate=0.1, 
    scale_pos_weight=pos_weight, # Dengesizliği çözen sihirli değnek
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

# 4. Tahminlerin Yapılması
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# 5. Raporlama
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
print("--- XGBOOST PERFORMANS RAPORU ---")
print(classification_report(y_test, y_pred))

# 6. MODELİ KAYDETME (İşte unuttuğumuz o kritik adım!)
# ---------------------------------------------------------
joblib.dump(xgb_model, "models/final_model.pkl")
print(" model 'models/final_model.pkl' olarak başarıyla kaydedildi!")


# ---------------------------------------------------------
# 7. GÖRSELLEŞTİRMELER
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Görsel 1: Karmaşıklık Matrisi
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[0])
axes[0].set_title('XGBoost Karmaşıklık Matrisi')
axes[0].set_xlabel('Tahmin Edilen')
axes[0].set_ylabel('Gerçek Değer')

# Görsel 2: ROC Eğrisi
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_title('XGBoost ROC Eğrisi')
axes[1].legend(loc="lower right")

# Görsel 3: Özellik Önem Derecesi (Feature Importance)
importances = pd.DataFrame({
    'Feature': X_train.columns, 
    'Importance': xgb_model.feature_importances_
})
top_features = importances.sort_values(by='Importance', ascending=False).head(10)

sns.barplot(x='Importance', y='Feature', data=top_features, ax=axes[2], palette='rocket')
axes[2].set_title('XGB En Önemli 10 Özellik')

plt.tight_layout()

plt.savefig("reports/xgb_results.png", dpi=300)
print("Grafik 'reports/xgb_results.png' olarak başarıyla kaydedildi!")
plt.show()


