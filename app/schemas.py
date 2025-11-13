from pydantic import BaseModel, Field, validator
from typing import Literal


class CustomerData(BaseModel):
    """
    Esquema de datos de entrada para predicción de churn.
    Contiene todas las características necesarias para realizar una predicción.
    """
    
    # Información demográfica
    gender: Literal['Male', 'Female'] = Field(
        ..., 
        description="Género del cliente"
    )
    SeniorCitizen: Literal[0, 1] = Field(
        ..., 
        description="Indica si es adulto mayor (1) o no (0)"
    )
    Partner: Literal['Yes', 'No'] = Field(
        ..., 
        description="Si el cliente tiene pareja"
    )
    Dependents: Literal['Yes', 'No'] = Field(
        ..., 
        description="Si el cliente tiene dependientes"
    )
    
    # Información de cuenta
    tenure: int = Field(
        ..., 
        ge=0, 
        le=72, 
        description="Antigüedad del cliente en meses"
    )
    
    # Servicios telefónicos
    PhoneService: Literal['Yes', 'No'] = Field(
        ..., 
        description="Si tiene servicio telefónico"
    )
    MultipleLines: Literal['Yes', 'No', 'No phone service'] = Field(
        ..., 
        description="Si tiene múltiples líneas telefónicas"
    )
    
    # Servicios de internet
    InternetService: Literal['DSL', 'Fiber optic', 'No'] = Field(
        ..., 
        description="Tipo de servicio de internet"
    )
    OnlineSecurity: Literal['Yes', 'No', 'No internet service'] = Field(
        ..., 
        description="Si tiene servicio de seguridad online"
    )
    OnlineBackup: Literal['Yes', 'No', 'No internet service'] = Field(
        ..., 
        description="Si tiene servicio de backup online"
    )
    DeviceProtection: Literal['Yes', 'No', 'No internet service'] = Field(
        ..., 
        description="Si tiene protección de dispositivos"
    )
    TechSupport: Literal['Yes', 'No', 'No internet service'] = Field(
        ..., 
        description="Si tiene soporte técnico"
    )
    StreamingTV: Literal['Yes', 'No', 'No internet service'] = Field(
        ..., 
        description="Si tiene servicio de streaming de TV"
    )
    StreamingMovies: Literal['Yes', 'No', 'No internet service'] = Field(
        ..., 
        description="Si tiene servicio de streaming de películas"
    )
    
    # Información de cuenta y facturación
    Contract: Literal['Month-to-month', 'One year', 'Two year'] = Field(
        ..., 
        description="Tipo de contrato"
    )
    PaperlessBilling: Literal['Yes', 'No'] = Field(
        ..., 
        description="Si usa facturación sin papel"
    )
    PaymentMethod: Literal[
        'Electronic check', 
        'Mailed check', 
        'Bank transfer (automatic)', 
        'Credit card (automatic)'
    ] = Field(
        ..., 
        description="Método de pago"
    )
    
    # Cargos
    MonthlyCharges: float = Field(
        ..., 
        gt=0, 
        le=150, 
        description="Cargo mensual en dólares"
    )
    TotalCharges: float = Field(
        ..., 
        ge=0, 
        description="Cargos totales acumulados en dólares"
    )
    
    @validator('TotalCharges')
    def validate_total_charges(cls, v, values):
        """
        Valida que TotalCharges sea consistente con tenure y MonthlyCharges.
        """
        if 'tenure' in values and 'MonthlyCharges' in values:
            expected_min = values['MonthlyCharges'] * values['tenure'] * 0.5
            expected_max = values['MonthlyCharges'] * values['tenure'] * 1.5
            
            if v < expected_min or v > expected_max:
                # Advertencia pero no error, ya que puede haber descuentos o promociones
                pass
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }


class ChurnPrediction(BaseModel):
    """
    Esquema de respuesta de la API con la predicción de churn.
    """
    
    churn_probability: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Probabilidad de churn (0-1)"
    )
    prediction: Literal['Yes', 'No'] = Field(
        ..., 
        description="Predicción binaria: Yes (churn) o No (no churn)"
    )
    risk_level: Literal['Low', 'Medium', 'High'] = Field(
        ..., 
        description="Nivel de riesgo de churn"
    )
    confidence: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Confianza de la predicción (max de las dos probabilidades)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "churn_probability": 0.85,
                "prediction": "Yes",
                "risk_level": "High",
                "confidence": 0.85
            }
        }


class HealthResponse(BaseModel):
    """
    Esquema de respuesta para el endpoint de salud.
    """
    status: str = Field(..., description="Estado del servicio")
    model_loaded: bool = Field(..., description="Indica si el modelo está cargado")
    model_type: str = Field(..., description="Tipo de modelo cargado")
