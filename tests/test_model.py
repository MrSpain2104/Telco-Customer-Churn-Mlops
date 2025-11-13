"""
Pruebas unitarias para el modelo de machine learning.
"""

import pytest
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


# Path al modelo
MODEL_PATH = Path(__file__).parent.parent / "app" / "model.joblib"


def test_model_exists():
    """
    Verifica que el archivo del modelo existe.
    """
    assert MODEL_PATH.exists(), f"El modelo no se encuentra en {MODEL_PATH}"


def test_model_can_load():
    """
    Verifica que el modelo se puede cargar correctamente.
    """
    try:
        model = joblib.load(MODEL_PATH)
        assert model is not None
    except Exception as e:
        pytest.fail(f"Error al cargar el modelo: {str(e)}")


def test_model_has_correct_methods():
    """
    Verifica que el modelo tiene los métodos necesarios.
    """
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, 'predict'), "El modelo no tiene método 'predict'"
    assert hasattr(model, 'predict_proba'), "El modelo no tiene método 'predict_proba'"


def test_model_prediction_shape():
    """
    Verifica que la forma de las predicciones es correcta.
    """
    model = joblib.load(MODEL_PATH)
    
    # Crear datos de ejemplo
    sample_data = pd.DataFrame({
        'gender': ['Female'],
        'SeniorCitizen': [0],
        'Partner': ['Yes'],
        'Dependents': ['No'],
        'tenure': [1],
        'PhoneService': ['No'],
        'MultipleLines': ['No phone service'],
        'InternetService': ['DSL'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['Yes'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['No'],
        'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [29.85],
        'TotalCharges': [29.85]
    })
    
    # Predicción
    prediction = model.predict(sample_data)
    proba = model.predict_proba(sample_data)
    
    # Verificar forma
    assert prediction.shape == (1,), f"Forma incorrecta de predicción: {prediction.shape}"
    assert proba.shape == (1, 2), f"Forma incorrecta de probabilidades: {proba.shape}"


def test_model_prediction_values():
    """
    Verifica que los valores de predicción son válidos.
    """
    model = joblib.load(MODEL_PATH)
    
    sample_data = pd.DataFrame({
        'gender': ['Female'],
        'SeniorCitizen': [0],
        'Partner': ['Yes'],
        'Dependents': ['No'],
        'tenure': [1],
        'PhoneService': ['No'],
        'MultipleLines': ['No phone service'],
        'InternetService': ['DSL'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['Yes'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['No'],
        'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [29.85],
        'TotalCharges': [29.85]
    })
    
    prediction = model.predict(sample_data)
    proba = model.predict_proba(sample_data)
    
    # Verificar que la predicción es 0 o 1
    assert prediction[0] in [0, 1], f"Predicción inválida: {prediction[0]}"
    
    # Verificar que las probabilidades suman 1
    assert np.isclose(proba.sum(), 1.0), f"Probabilidades no suman 1: {proba.sum()}"
    
    # Verificar que las probabilidades están en [0, 1]
    assert (proba >= 0).all() and (proba <= 1).all(), "Probabilidades fuera de rango [0, 1]"


def test_model_prediction_consistency():
    """
    Verifica que el modelo da predicciones consistentes para la misma entrada.
    """
    model = joblib.load(MODEL_PATH)
    
    sample_data = pd.DataFrame({
        'gender': ['Male'],
        'SeniorCitizen': [1],
        'Partner': ['No'],
        'Dependents': ['No'],
        'tenure': [60],
        'PhoneService': ['Yes'],
        'MultipleLines': ['Yes'],
        'InternetService': ['Fiber optic'],
        'OnlineSecurity': ['Yes'],
        'OnlineBackup': ['Yes'],
        'DeviceProtection': ['Yes'],
        'TechSupport': ['Yes'],
        'StreamingTV': ['Yes'],
        'StreamingMovies': ['Yes'],
        'Contract': ['Two year'],
        'PaperlessBilling': ['No'],
        'PaymentMethod': ['Bank transfer (automatic)'],
        'MonthlyCharges': [105.50],
        'TotalCharges': [6330.00]
    })
    
    # Hacer múltiples predicciones
    pred1 = model.predict_proba(sample_data)
    pred2 = model.predict_proba(sample_data)
    pred3 = model.predict_proba(sample_data)
    
    # Verificar que son iguales
    assert np.allclose(pred1, pred2), "Predicciones inconsistentes"
    assert np.allclose(pred2, pred3), "Predicciones inconsistentes"


def test_model_handles_edge_cases():
    """
    Verifica que el modelo maneja casos extremos correctamente.
    """
    model = joblib.load(MODEL_PATH)
    
    # Cliente muy nuevo con contrato flexible (alto riesgo)
    high_risk_customer = pd.DataFrame({
        'gender': ['Female'],
        'SeniorCitizen': [0],
        'Partner': ['No'],
        'Dependents': ['No'],
        'tenure': [0],
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
        'InternetService': ['Fiber optic'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['No'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['No'],
        'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [100.00],
        'TotalCharges': [0.00]
    })
    
    # Cliente muy antiguo con contrato largo (bajo riesgo)
    low_risk_customer = pd.DataFrame({
        'gender': ['Male'],
        'SeniorCitizen': [0],
        'Partner': ['Yes'],
        'Dependents': ['Yes'],
        'tenure': [72],
        'PhoneService': ['Yes'],
        'MultipleLines': ['Yes'],
        'InternetService': ['DSL'],
        'OnlineSecurity': ['Yes'],
        'OnlineBackup': ['Yes'],
        'DeviceProtection': ['Yes'],
        'TechSupport': ['Yes'],
        'StreamingTV': ['Yes'],
        'StreamingMovies': ['Yes'],
        'Contract': ['Two year'],
        'PaperlessBilling': ['No'],
        'PaymentMethod': ['Bank transfer (automatic)'],
        'MonthlyCharges': [50.00],
        'TotalCharges': [3600.00]
    })
    
    # Predicciones
    high_risk_proba = model.predict_proba(high_risk_customer)[0][1]
    low_risk_proba = model.predict_proba(low_risk_customer)[0][1]
    
    # El cliente de alto riesgo debería tener mayor probabilidad de churn
    assert high_risk_proba > low_risk_proba, \
        f"Lógica de predicción incorrecta: alto riesgo ({high_risk_proba:.4f}) " \
        f"<= bajo riesgo ({low_risk_proba:.4f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
