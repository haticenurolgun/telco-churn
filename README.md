telco-churn-prediction/
│
├── data/                       # Veri setleri (Git'e yüklenmemeli!)
│   ├── raw/                    # Kaggle'dan indirilen ham veri (Telco-Customer-Churn.csv)
│   └── processed/              # Ön işlemden geçmiş, modele hazır temiz veri
│
├── notebooks/                  # Veri analizi ve ön işleme çalışmaları
│   ├── 01_eda_analizi.ipynb    # 1. Kişinin (EDA) çalışma alanı
│   └── 02_on_isleme.ipynb      # 2. Kişinin (Veri Ön İşleme) çalışma alanı
│
├── src/                        # Makine Öğrenmesi kaynak kodları
│   ├── train.py                # 3. Kişinin modeli eğiteceği ana script
│   └── utils.py                # Ortak yardımcı fonksiyonlar
│
├── app/                        # API servis dosyaları
│   ├── main.py                 # 4. Kişinin yazacağı Flask/FastAPI uygulaması
│   └── schemas.py              # (FastAPI kullanılacaksa) Pydantic veri modelleri
│
├── models/                     # Eğitilmiş model dosyaları
│   └── final_model.pkl         # Eğitilen modelin API için dışa aktarılmış hali
│
├── .gitignore                  # Git'in takip etmemesi gereken dosyalar listesi
├── requirements.txt            # Ortak kütüphane listesi (pandas, scikit-learn vb.)
└── README.md                   # Proje vitrini ve kurulum talimatları


# 📞 Telco Customer Churn Prediction

Bu proje, telekomünikasyon müşterilerinin davranış verilerini analiz ederek, şirketi terk etme (churn) ihtimallerini makine öğrenmesi teknikleriyle önceden tahmin etmeyi amaçlamaktadır.

## 📂 Proje Klasör Yapısı

Projeye ait dosyalar modüler ve ölçeklenebilir bir yapıda organize edilmiştir:

* **`data/`**: Veri setinin ham (`raw/`) ve eğitim için temizlenmiş (`processed/`) versiyonlarını barındırır.
* **`models/`**: Eğitilmiş ve API tarafından tüketilmeye hazır makine öğrenmesi modelini (`final_model.pkl`) içerir.
* **`notebooks/`**: Keşifsel Veri Analizi (EDA) ve veri ön işleme adımlarının adım adım dökümünü içerir.
* **`reports/`**: Modellerin performans metriklerini, karmaşıklık matrislerini ve ROC eğrisi grafiklerini tutar.
* **`src/`**: Model eğitimlerini (`train_logreg.py`, `train_rf.py`, `train_xgb.py`) gerçekleştiren kaynak kodları barındırır.
* **`app/`**: (Geliştirme Aşamasında) Tahmin modelini dış dünyaya açacak API kodlarını içerir.

## 📊 Model Performansı

Proje kapsamında Lojistik Regresyon, Random Forest ve XGBoost algoritmaları denenmiştir. Dengesiz veri setindeki "Churn (1)" sınıfını yakalama başarısı (Recall) ve genel dengesi göz önüne alınarak **XGBoost** şampiyon model olarak seçilmiştir.

* **Model:** XGBoost Classifier
* **Recall (Sınıf 1):** %77
* **F1-Score:** %0.60
* *Dengesiz veri seti problemine karşı `scale_pos_weight` parametresi ile optimizasyon yapılmıştır.*

## 🚀 Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1.  Repoyu klonlayın ve sanal ortamınızı oluşturun.
2.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install -r requirements.txt
    ```
3.  Şampiyon modeli tekrar eğitmek ve grafikleri oluşturmak için:
    ```bash
    python src/train_xgb.py
    ```

## 🔌 API Ekibi İçin Notlar

Model eğitimi tamamlanmış ve `models/final_model.pkl` yoluna kaydedilmiştir. API üzerinden modele veri gönderilirken, gelen verinin sütun yapısı ve sıralamasının `data/processed/X_train.csv` dosyası ile **birebir aynı** olduğundan emin olunmalıdır. Encoding (One-Hot) işlemleri dışarıdan gelen JSON verisine uygulanmalıdır.