from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI(
    title="API Predicción Churn - Banco Alura",
    description="Predice la probabilidad de abandono",
    version="1.0"
)
# ---------------------------------
# Cargar modelo
# ---------------------------------
pipe_xgb = joblib.load("modelo_Banco_Alura_churn.pkl")

app = FastAPI(
    title="API Predicción Churn - Banco Alura",
    description="Predice la probabilidad de abandono de clientes",
    version="1.0"
)

# ---------------------------------
# Esquema de entrada
# ---------------------------------
class ClienteInput(BaseModel):
    Age: int = Field(..., example=40, description="Edad del cliente")
    NumOfProducts: int = Field(..., example=1, description="Número de productos contratados")
    Inactivo_40_70: int = Field(..., example=0, description="Indicador de inactividad (1: Sí, 0: No)")
    Products_Risk_Flag: int = Field(..., example=0, description="Bandera de riesgo por productos")
    Country_Risk_Flag: int = Field(..., example=1, description="Bandera de riesgo por país")
    
# ---------------------------------
# Endpoint de predicción
# ---------------------------------
@app.post("/predict")
def predict_churn(cliente: ClienteInput):
    # .model_dump() es el estándar en Pydantic v2
    data = pd.DataFrame([cliente.model_dump()])
    
    # El pipeline ya debe tener el escalado/transformación incluido
    prob = pipe_xgb.predict_proba(data)[0, 1]

    return {
        "probabilidad_abandono": round(float(prob), 4),
        "abandona": bool(prob >= 0.58)
    }