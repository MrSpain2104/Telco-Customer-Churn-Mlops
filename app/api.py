"""
API REST con FastAPI para predicción de churn en telecomunicaciones.

Este servicio permite realizar predicciones en tiempo real utilizando
el modelo entrenado de machine learning.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any

from .schemas import CustomerData, ChurnPrediction, HealthResponse

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Telco Churn Prediction API",
    description="API para predecir la probabilidad de churn de clientes de telecomunicaciones",
    version="1.0.0",
    contact={
        "name": "Data Science Team",
        "email": "support@telco-churn.com"
    }
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
MODEL = None
MODEL_PATH = Path(__file__).parent / "model.joblib"
MODEL_TYPE = None


def load_model():
    """
    Carga el modelo entrenado desde disco.
    """
    global MODEL, MODEL_TYPE
    
    try:
        logger.info(f"Cargando modelo desde: {MODEL_PATH}")
        MODEL = joblib.load(MODEL_PATH)
        
        # Detectar tipo de modelo
        if hasattr(MODEL, 'named_steps'):
            classifier = MODEL.named_steps.get('classifier')
            if classifier:
                MODEL_TYPE = type(classifier).__name__
            else:
                MODEL_TYPE = "Unknown Pipeline"
        else:
            MODEL_TYPE = type(MODEL).__name__
        
        logger.info(f" Modelo cargado exitosamente: {MODEL_TYPE}")
        return True
        
    except FileNotFoundError:
        logger.error(f" No se encontró el archivo del modelo en: {MODEL_PATH}")
        logger.error("   Asegúrate de entrenar el modelo primero (ejecutar notebook 2)")
        return False
    except Exception as e:
        logger.error(f" Error al cargar el modelo: {str(e)}")
        return False


def get_risk_level(probability: float) -> str:
    """
    Determina el nivel de riesgo basado en la probabilidad de churn.
    
    Args:
        probability: Probabilidad de churn (0-1)
    
    Returns:
        str: 'Low', 'Medium', o 'High'
    """
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"


# Eventos de inicio y cierre
@app.on_event("startup")
async def startup_event():
    """
    Se ejecuta al iniciar la aplicación.
    Carga el modelo en memoria.
    """
    logger.info("=" * 80)
    logger.info("Iniciando Telco Churn Prediction API")
    logger.info("=" * 80)
    
    success = load_model()
    
    if not success:
        logger.warning("El servicio se inició sin un modelo cargado")
        logger.warning("La API funcionará pero las predicciones fallarán")
    else:
        logger.info("Servicio iniciado correctamente")
    
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Se ejecuta al cerrar la aplicación.
    """
    logger.info("Cerrando Telco Churn Prediction API")


# Endpoints
@app.get("/", tags=["General"])
async def root():
    """
    Endpoint raíz - Información básica de la API.
    """
    return {
        "message": "Telco Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Endpoint de salud - Verifica que el servicio esté funcionando correctamente.
    
    Returns:
        HealthResponse: Estado del servicio y del modelo.
    """
    return HealthResponse(
        status="healthy" if MODEL is not None else "degraded",
        model_loaded=MODEL is not None,
        model_type=MODEL_TYPE if MODEL_TYPE else "No model loaded"
    )


@app.post("/predict", response_model=ChurnPrediction, tags=["Predictions"])
async def predict_churn(customer: CustomerData):
    """
    Predice la probabilidad de churn para un cliente.
    
    Args:
        customer: Datos del cliente (CustomerData schema)
    
    Returns:
        ChurnPrediction: Predicción con probabilidad, clasificación y nivel de riesgo.
    
    Raises:
        HTTPException: Si el modelo no está cargado o hay un error en la predicción.
    """
    # Verificar que el modelo esté cargado
    if MODEL is None:
        logger.error("Intento de predicción sin modelo cargado")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo no está cargado. Por favor, contacte al administrador."
        )
    
    try:
        # Convertir datos de entrada a DataFrame
        input_data = pd.DataFrame([customer.dict()])
        
        logger.info(f"Predicción solicitada para cliente con tenure={customer.tenure}, "
                   f"Contract={customer.Contract}, MonthlyCharges={customer.MonthlyCharges}")
        
        # Realizar predicción
        prediction_proba = MODEL.predict_proba(input_data)
        churn_probability = float(prediction_proba[0][1])
        
        # Determinar predicción binaria
        prediction_binary = "Yes" if churn_probability > 0.5 else "No"
        
        # Calcular confianza (máximo de las dos probabilidades)
        confidence = float(max(prediction_proba[0]))
        
        # Determinar nivel de riesgo
        risk = get_risk_level(churn_probability)
        
        logger.info(f"Predicción exitosa: Churn={prediction_binary}, "
                   f"Probabilidad={churn_probability:.4f}, Riesgo={risk}")
        
        return ChurnPrediction(
            churn_probability=churn_probability,
            prediction=prediction_binary,
            risk_level=risk,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al realizar la predicción: {str(e)}"
        )


@app.post("/predict-batch", tags=["Predictions"])
async def predict_batch(customers: list[CustomerData]):
    """
    Predice la probabilidad de churn para múltiples clientes.
    
    Args:
        customers: Lista de datos de clientes
    
    Returns:
        dict: Predicciones para cada cliente
    """
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo no está cargado."
        )
    
    try:
        predictions = []
        
        for idx, customer in enumerate(customers):
            input_data = pd.DataFrame([customer.dict()])
            prediction_proba = MODEL.predict_proba(input_data)
            churn_probability = float(prediction_proba[0][1])
            
            predictions.append({
                "customer_index": idx,
                "churn_probability": churn_probability,
                "prediction": "Yes" if churn_probability > 0.5 else "No",
                "risk_level": get_risk_level(churn_probability),
                "confidence": float(max(prediction_proba[0]))
            })
        
        logger.info(f"Predicción batch exitosa: {len(predictions)} clientes procesados")
        
        return {
            "total_customers": len(customers),
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f" Error en predicción batch: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al realizar la predicción batch: {str(e)}"
        )


@app.get("/model-info", tags=["Model"])
async def get_model_info():
    """
    Obtiene información sobre el modelo cargado.
    
    Returns:
        dict: Información del modelo
    """
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo no está cargado."
        )
    
    try:
        info = {
            "model_type": MODEL_TYPE,
            "model_path": str(MODEL_PATH),
            "pipeline_steps": list(MODEL.named_steps.keys()) if hasattr(MODEL, 'named_steps') else [],
        }
        
        # Intentar obtener información adicional del clasificador
        if hasattr(MODEL, 'named_steps'):
            classifier = MODEL.named_steps.get('classifier')
            if classifier:
                if hasattr(classifier, 'n_estimators'):
                    info['n_estimators'] = classifier.n_estimators
                if hasattr(classifier, 'max_depth'):
                    info['max_depth'] = classifier.max_depth
                if hasattr(classifier, 'learning_rate'):
                    info['learning_rate'] = classifier.learning_rate
        
        return info
        
    except Exception as e:
        logger.error(f"Error al obtener información del modelo: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener información del modelo: {str(e)}"
        )


# Manejador de errores globales
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Manejador global de excepciones.
    """
    logger.error(f"Error no manejado: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Ocurrió un error interno en el servidor.",
            "error": str(exc)
        }
    )


# Punto de entrada para desarrollo local
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Iniciando servidor en modo desarrollo...")
    logger.info("Documentación disponible en: http://localhost:8000/docs")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
