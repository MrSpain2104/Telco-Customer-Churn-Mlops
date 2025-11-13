"""
Script para probar la API localmente de forma rápida.
"""

import requests
import json

# URL base de la API
BASE_URL = "http://localhost:8000"


def test_health():
    """Prueba el endpoint de salud."""
    print("\n" + "="*80)
    print("TEST: Endpoint de Salud")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_prediction():
    """Prueba el endpoint de predicción."""
    print("\n" + "="*80)
    print("TEST: Predicción Individual")
    print("="*80)
    
    # Cliente con alto riesgo de churn
    customer_data = {
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
    
    print("\nDatos del cliente (alto riesgo):")
    print(f"  - tenure: {customer_data['tenure']} meses")
    print(f"  - Contract: {customer_data['Contract']}")
    print(f"  - PaymentMethod: {customer_data['PaymentMethod']}")
    print(f"  - MonthlyCharges: ${customer_data['MonthlyCharges']}")
    
    response = requests.post(f"{BASE_URL}/predict", json=customer_data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_prediction_low_risk():
    """Prueba con un cliente de bajo riesgo."""
    print("\n" + "="*80)
    print("TEST: Predicción Individual (Bajo Riesgo)")
    print("="*80)
    
    # Cliente con bajo riesgo de churn
    customer_data = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
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
    
    print("\nDatos del cliente (bajo riesgo):")
    print(f"  - tenure: {customer_data['tenure']} meses")
    print(f"  - Contract: {customer_data['Contract']}")
    print(f"  - PaymentMethod: {customer_data['PaymentMethod']}")
    print(f"  - MonthlyCharges: ${customer_data['MonthlyCharges']}")
    
    response = requests.post(f"{BASE_URL}/predict", json=customer_data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_batch_prediction():
    """Prueba el endpoint de predicción batch."""
    print("\n" + "="*80)
    print("TEST: Predicción Batch (2 clientes)")
    print("="*80)
    
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
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
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
    
    response = requests.post(f"{BASE_URL}/predict-batch", json=customers)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_model_info():
    """Prueba el endpoint de información del modelo."""
    print("\n" + "="*80)
    print("TEST: Información del Modelo")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SUITE DE PRUEBAS DE LA API - TELCO CHURN PREDICTION")
    print("="*80)
    print(f"\nURL Base: {BASE_URL}")
    print("\nAsegúrate de que la API esté corriendo:")
    print("  uvicorn app.api:app --reload")
    
    try:
        # Ejecutar todas las pruebas
        test_health()
        test_prediction()
        test_prediction_low_risk()
        test_batch_prediction()
        test_model_info()
        
        print("\n" + "="*80)
        print("✅ TODAS LAS PRUEBAS COMPLETADAS")
        print("="*80 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: No se pudo conectar a la API")
        print("   Verifica que la API esté corriendo en http://localhost:8000")
        print("   Ejecuta: uvicorn app.api:app --reload\n")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")
