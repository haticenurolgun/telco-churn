from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import xgboost

app = FastAPI(
    title="Müşteri Kaybı (Churn) Tahmin API",
    description="Telekom müşterilerinin churn olasılığını tahmin eden sistem.",
    version="1.0.0",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}  
)

model = joblib.load("final_model.pkl")

class CustomerData(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    NewCustomer: int
    AvgCharges: float
    gender_Male: int
    Partner_Yes: int
    Dependents_Yes: int
    PhoneService_Yes: int
    MultipleLines_No_phone_service: int = Field(alias="MultipleLines_No phone service")
    MultipleLines_Yes: int
    InternetService_Fiber_optic: int = Field(alias="InternetService_Fiber optic")
    InternetService_No: int
    OnlineSecurity_No_internet_service: int = Field(alias="OnlineSecurity_No internet service")
    OnlineSecurity_Yes: int
    OnlineBackup_No_internet_service: int = Field(alias="OnlineBackup_No internet service")
    OnlineBackup_Yes: int
    DeviceProtection_No_internet_service: int = Field(alias="DeviceProtection_No internet service")
    DeviceProtection_Yes: int
    TechSupport_No_internet_service: int = Field(alias="TechSupport_No internet service")
    TechSupport_Yes: int
    StreamingTV_No_internet_service: int = Field(alias="StreamingTV_No internet service")
    StreamingTV_Yes: int
    StreamingMovies_No_internet_service: int = Field(alias="StreamingMovies_No internet service")
    StreamingMovies_Yes: int
    Contract_One_year: int = Field(alias="Contract_One year")
    Contract_Two_year: int = Field(alias="Contract_Two year")
    PaperlessBilling_Yes: int
    PaymentMethod_Credit_card_automatic: int = Field(alias="PaymentMethod_Credit card (automatic)")
    PaymentMethod_Electronic_check: int = Field(alias="PaymentMethod_Electronic check")
    PaymentMethod_Mailed_check: int = Field(alias="PaymentMethod_Mailed check")

@app.post("/predict")
def predict_churn(customer: CustomerData):
    
    input_data = customer.model_dump(by_alias=True)
    df = pd.DataFrame([input_data])
    
    prediction = model.predict(df)
    probability = model.predict_proba(df)[0][1]
    
    # Müşteri iptal eder mi? (1: Eder, 0: Etmez)
    sonuc_metni = "Müşteri iptal edebilir (CHURN)" if prediction[0] == 1 else "Müşteri kalıcı (KALIR)"
    
    return {
        "tahmin_kodu": int(prediction[0]),
        "tahmin_metni": sonuc_metni,
        "iptal_etme_olasiligi": f"%{round(float(probability) * 100, 2)}",
        "mesaj": "Sistem başarıyla çalıştı."
    }