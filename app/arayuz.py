import streamlit as st
import requests

st.set_page_config(page_title="Müşteri Kaybı Tahmini", layout="wide")

st.title("🚀 Tam Kapsamlı Müşteri Kaybı (Churn) Tahmin Paneli")
st.write("Lütfen müşterinin tüm detaylarını girin. Model bu 32 veriyi analiz ederek karar verecektir.")
st.markdown("---")

cevir = {"Hayır": 0, "Evet": 1}

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Profil ve Faturalandırma")
    senior = st.selectbox("Yaşlı Müşteri mi? ", ["Hayır", "Evet"])
    gender = st.selectbox("Cinsiyet Erkek mi? ", ["Evet", "Hayır"])
    partner = st.selectbox("Evli/Partneri var mı? ", ["Evet", "Hayır"])
    dep = st.selectbox("Bakmakla Yükümlü Olduğu Biri Var mı? ", ["Hayır", "Evet"])
    new_cust = st.selectbox("Yeni Müşteri mi? ", ["Hayır", "Evet"])
    
    st.markdown("---")
    tenure = st.number_input("Müşterilik Süresi (Ay)", min_value=0, value=12)
    monthly = st.number_input("Aylık Fatura (TL)", min_value=0.0, value=75.5)
    total = st.number_input("Toplam Harcama (TL)", min_value=0.0, value=906.0)
    avg = st.number_input("Ortalama Fatura (TL)", min_value=0.0, value=75.5)

with col2:
    st.subheader("🌐 Temel Servisler")
    phone = st.selectbox("Telefon Hizmeti Var mı?", ["Evet", "Hayır"])
    mult_yes = st.selectbox("Çoklu Hat Var mı?", ["Hayır", "Evet"])
    mult_no_phone = st.selectbox("Çoklu Hat (Telefon Hizmeti Yok durumu)", ["Hayır", "Evet"])
    
    st.markdown("---")
    int_fiber = st.selectbox("İnternet: Fiber Optik mi?", ["Evet", "Hayır"])
    int_no = st.selectbox("İnternet Hizmeti Yok mu?", ["Hayır", "Evet"])
    sec_yes = st.selectbox("Online Güvenlik Var mı?", ["Hayır", "Evet"])
    sec_no_int = st.selectbox("Online Güvenlik (İnternet Yok durumu)", ["Hayır", "Evet"])
    back_yes = st.selectbox("Online Yedekleme Var mı?", ["Hayır", "Evet"])
    back_no_int = st.selectbox("Online Yedekleme (İnternet Yok durumu)", ["Hayır", "Evet"])

with col3:
    st.subheader("🛡️ Ek Servisler ve Sözleşme")
    dev_yes = st.selectbox("Cihaz Koruma Var mı?", ["Hayır", "Evet"])
    dev_no_int = st.selectbox("Cihaz Koruma (İnternet Yok durumu)", ["Hayır", "Evet"])
    tech_yes = st.selectbox("Teknik Destek Var mı?", ["Hayır", "Evet"])
    tech_no_int = st.selectbox("Teknik Destek (İnternet Yok durumu)", ["Hayır", "Evet"])
    tv_yes = st.selectbox("Streaming TV Var mı?", ["Hayır", "Evet"])
    tv_no_int = st.selectbox("Streaming TV (İnternet Yok durumu)", ["Hayır", "Evet"])
    mov_yes = st.selectbox("Streaming Filmler Var mı?", ["Hayır", "Evet"])
    mov_no_int = st.selectbox("Streaming Filmler (İnternet Yok durumu)", ["Hayır", "Evet"])
    
    st.markdown("---")
    contract_1 = st.selectbox("Sözleşme: 1 Yıllık mı?", ["Hayır", "Evet"])
    contract_2 = st.selectbox("Sözleşme: 2 Yıllık mı?", ["Hayır", "Evet"])
    paperless = st.selectbox("Kağıtsız Fatura mı?", ["Evet", "Hayır"])
    pay_credit = st.selectbox("Ödeme: Kredi Kartı ile mi?", ["Hayır", "Evet"])
    pay_elec = st.selectbox("Ödeme: Elektronik Çek ile mi?", ["Evet", "Hayır"])
    pay_mail = st.selectbox("Ödeme: Posta Çeki ile mi?", ["Hayır", "Evet"])

st.markdown("---")


if st.button("🔮 Tahmini Gör (32 Veriyi Analiz Et)", use_container_width=True):
    
    payload = {
        "SeniorCitizen": cevir[senior],
        "tenure": int(tenure),
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total),
        "NewCustomer": cevir[new_cust],
        "AvgCharges": float(avg),
        "gender_Male": cevir[gender],
        "Partner_Yes": cevir[partner],
        "Dependents_Yes": cevir[dep],
        "PhoneService_Yes": cevir[phone],
        "MultipleLines_No phone service": cevir[mult_no_phone],
        "MultipleLines_Yes": cevir[mult_yes],
        "InternetService_Fiber optic": cevir[int_fiber],
        "InternetService_No": cevir[int_no],
        "OnlineSecurity_No internet service": cevir[sec_no_int],
        "OnlineSecurity_Yes": cevir[sec_yes],
        "OnlineBackup_No internet service": cevir[back_no_int],
        "OnlineBackup_Yes": cevir[back_yes],
        "DeviceProtection_No internet service": cevir[dev_no_int],
        "DeviceProtection_Yes": cevir[dev_yes],
        "TechSupport_No internet service": cevir[tech_no_int],
        "TechSupport_Yes": cevir[tech_yes],
        "StreamingTV_No internet service": cevir[tv_no_int],
        "StreamingTV_Yes": cevir[tv_yes],
        "StreamingMovies_No internet service": cevir[mov_no_int],
        "StreamingMovies_Yes": cevir[mov_yes],
        "Contract_One year": cevir[contract_1],
        "Contract_Two year": cevir[contract_2],
        "PaperlessBilling_Yes": cevir[paperless],
        "PaymentMethod_Credit card (automatic)": cevir[pay_credit],
        "PaymentMethod_Electronic check": cevir[pay_elec],
        "PaymentMethod_Mailed check": cevir[pay_mail]
    }
    
    try:
        
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            sonuc = response.json()
            if sonuc["tahmin_kodu"] == 1:
                st.error(f"⚠️ DİKKAT: Bu müşteri aboneliğini İPTAL EDEBİLİR! (Olasılık: {sonuc['iptal_etme_olasiligi']})")
            else:
                st.success(f"✅ GÜVENDE: Bu müşteri sistemde KALACAK. (İptal Olasılığı: {sonuc['iptal_etme_olasiligi']})")
        else:
            st.warning("API veri tiplerinde bir hata tespit etti.")
            
    except requests.exceptions.ConnectionError:
        st.error("Bağlantı Hatası! Lütfen API sunucusunun (uvicorn) açık olduğundan emin olun.")