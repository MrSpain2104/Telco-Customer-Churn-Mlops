
import requests
import json

# URL de la API
API_URL = "http://localhost:8001"

# Ejemplo 1: Cliente con ALTO riesgo de churn
print("\n" + "="*80)
print("EJEMPLO 1: Cliente con ALTO RIESGO de Churn")
print("="*80)

cliente_alto_riesgo = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,  # Cliente nuevo
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",  # Sin servicios adicionales
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",  # Contrato flexible
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",  # Método de pago riesgoso
    "MonthlyCharges": 85.5,  # Cargos altos
    "TotalCharges": 256.5
}

response = requests.post(f"{API_URL}/predict", json=cliente_alto_riesgo)
resultado = response.json()

print(f"\nResultado de la predicción:")
print(f"   Predicción: {resultado['prediction']}")
print(f"   Probabilidad de Churn: {resultado['churn_probability']:.2%}")
print(f"   Nivel de Riesgo: {resultado['risk_level']}")
print(f"   Confianza: {resultado['confidence']:.2%}")

# Ejemplo 2: Cliente con BAJO riesgo de churn
print("\n" + "="*80)
print("EJEMPLO 2: Cliente con BAJO RIESGO de Churn")
print("="*80)

cliente_bajo_riesgo = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 65,  # Cliente leal
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",  # Con servicios adicionales
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Two year",  # Contrato largo plazo
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",  # Pago automático
    "MonthlyCharges": 45.0,  # Cargos moderados
    "TotalCharges": 2925.0
}

response = requests.post(f"{API_URL}/predict", json=cliente_bajo_riesgo)
resultado = response.json()

print(f"\nResultado de la predicción:")
print(f"   Predicción: {resultado['prediction']}")
print(f"   Probabilidad de Churn: {resultado['churn_probability']:.2%}")
print(f"   Nivel de Riesgo: {resultado['risk_level']}")
print(f"   Confianza: {resultado['confidence']:.2%}")

# Ejemplo 3: Predicción por lotes (múltiples clientes)
print("\n" + "="*80)
print("EJEMPLO 3: Predicción por Lotes (Batch)")
print("="*80)

clientes_lote = [cliente_alto_riesgo, cliente_bajo_riesgo]

response = requests.post(f"{API_URL}/predict-batch", json={"customers": clientes_lote})
resultados = response.json()

print(f"\nResultados de {len(resultados['predictions'])} predicciones:")
for i, pred in enumerate(resultados['predictions'], 1):
    print(f"\n   Cliente {i}:")
    print(f"      Predicción: {pred['prediction']}")
    print(f"      Probabilidad: {pred['churn_probability']:.2%}")
    print(f"      Riesgo: {pred['risk_level']}")

# Ejemplo 4: Información del modelo
print("\n" + "="*80)
print("EJEMPLO 4: Información del Modelo")
print("="*80)

response = requests.get(f"{API_URL}/model-info")
info = response.json()

print(f"\nInformación del modelo:")
print(f"   Tipo: {info['model_type']}")
print(f"   Características: {info['n_features']}")
print(f"   Fecha de carga: {info['loaded_at']}")

print("\n" + "="*80)
print("¡Todos los ejemplos ejecutados exitosamente!")
print("="*80)
