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