
import requests
import json

API_URL = "http://localhost:8001"

def test_health():
    """Probar endpoint de salud"""
    print("\n1. Probando /health...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_prediction():
    """Probar predicción"""
    print("\n2. Probando /predict...")
    
    # Cliente de alto riesgo
    customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 256.5
    }
    
    try:
        response = requests.post(f"{API_URL}/predict", json=customer, timeout=5)
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Prediction: {result['prediction']}")
        print(f"   Probability: {result['churn_probability']:.4f}")
        print(f"   Risk Level: {result['risk_level']}")
        return response.status_code == 200
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_model_info():
    """Probar información del modelo"""
    print("\n3. Probando /model-info...")
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        print(f"   Status: {response.status_code}")
        info = response.json()
        print(f"   Model: {info['model_type']}")
        print(f"   Features: {info['n_features']}")
        return response.status_code == 200
    except Exception as e:
        print(f"   Error: {e}")
        return False

if __name__ == "__main__":
    print("="*80)
    print("PRUEBA RÁPIDA DE LA API")
    print("="*80)
    print(f"\nAsegúrate de que la API esté corriendo en {API_URL}")
    print("Comando: uvicorn app.api:app --reload --port 8001\n")
    
    input("Presiona Enter cuando la API esté lista...")
    
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Prediction", test_prediction()))
    results.append(("Model Info", test_model_info()))
    
    print("\n" + "="*80)
    print("RESUMEN DE PRUEBAS")
    print("="*80)
    for test_name, passed in results:
        status = " PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n ¡Todos los tests pasaron! La API está funcionando correctamente.")
    else:
        print("\n Algunos tests fallaron. Revisa la configuración de la API.")
