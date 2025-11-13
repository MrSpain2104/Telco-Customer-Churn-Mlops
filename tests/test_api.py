import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Añadir el directorio app al path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from api import app

client = TestClient(app)


def test_root_endpoint():
    """
    Test del endpoint raíz.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["status"] == "running"


def test_health_endpoint():
    """
    Test del endpoint de salud.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_type" in data


def test_predict_endpoint_valid_data():
    """
    Test del endpoint de predicción con datos válidos.
    """
    valid_customer = {
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
    
    response = client.post("/predict", json=valid_customer)
    
    # Si el modelo está cargado, debería retornar 200
    # Si no está cargado, debería retornar 503
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "churn_probability" in data
        assert "prediction" in data
        assert "risk_level" in data
        assert "confidence" in data
        assert 0 <= data["churn_probability"] <= 1
        assert data["prediction"] in ["Yes", "No"]
        assert data["risk_level"] in ["Low", "Medium", "High"]


def test_predict_endpoint_invalid_data():
    """
    Test del endpoint de predicción con datos inválidos.
    """
    invalid_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": -5,  # Inválido: tenure negativo
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
    
    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422  # Unprocessable Entity (validación fallida)


def test_predict_endpoint_missing_fields():
    """
    Test del endpoint de predicción con campos faltantes.
    """
    incomplete_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes"
        # Faltan muchos campos requeridos
    }
    
    response = client.post("/predict", json=incomplete_customer)
    assert response.status_code == 422


def test_predict_batch_endpoint():
    """
    Test del endpoint de predicción batch.
    """
    customers = [
        {
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
        },
        {
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 60,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 105.50,
            "TotalCharges": 6330.00
        }
    ]
    
    response = client.post("/predict-batch", json=customers)
    
    # Si el modelo está cargado, debería retornar 200
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "total_customers" in data
        assert "predictions" in data
        assert data["total_customers"] == 2
        assert len(data["predictions"]) == 2


def test_model_info_endpoint():
    """
    Test del endpoint de información del modelo.
    """
    response = client.get("/model-info")
    
    # Si el modelo está cargado, debería retornar 200
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "model_type" in data


def test_risk_level_categorization():
    """
    Test de la categorización de niveles de riesgo.
    """
    from api import get_risk_level
    
    assert get_risk_level(0.1) == "Low"
    assert get_risk_level(0.29) == "Low"
    assert get_risk_level(0.3) == "Medium"
    assert get_risk_level(0.5) == "Medium"
    assert get_risk_level(0.69) == "Medium"
    assert get_risk_level(0.7) == "High"
    assert get_risk_level(0.95) == "High"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
