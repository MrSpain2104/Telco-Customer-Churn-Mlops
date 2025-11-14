# Telco Customer Churn Prediction - MLOps Implementation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

## Equipo de Desarrollo

**Universidad del Norte - Curso de Machine Learning**

- **Omar Medina** 
- **Andrés España** 
---

##  Descripción

Proyecto de **Machine Learning** y **MLOps** para predecir el abandono (churn) de clientes en una empresa de telecomunicaciones. Incluye análisis exploratorio, entrenamiento de modelos, interpretabilidad con LIME y despliegue con FastAPI y Docker.

### Características Principales

-  **Análisis Exploratorio de Datos (EDA)** completo
-  **4 Modelos de ML**: Random Forest, XGBoost, CatBoost, LightGBM
-  **Optimización de hiperparámetros** con GridSearchCV
-  **Interpretabilidad** con LIME
-  **API REST** con FastAPI
-  **Containerización** con Docker
-  **CI/CD** con GitHub Actions
-  **Tests unitarios** con pytest

---

## Dataset

**Telco Customer Churn** de Kaggle
- **Tamaño**: ~7,000 registros
- **Características**: 20 variables (demográficas, servicios, facturación)
- **Variable objetivo**: Churn (Yes/No)
- **Tasa de churn**: ~27%

**Enlace**: [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## Estructura del Proyecto

```
Telco-Customer-Churn-Mlops/
├── app/                           # Servicio FastAPI con el modelo
│   ├── api.py
│   ├── schemas.py
│   ├── model.joblib
│   └── __init__.py
├── data/                          # Datasets originales y limpios
├── notebooks/                     # Notebooks 
│   ├── 1_eda_preprocessing.ipynb
│   ├── 2_model_training.ipynb
│   └── 3_interpretability.ipynb
├── jupyter-book/                  # Documentación en Jupyter Book
│   ├── intro.md
│   ├── notebooks/
│   ├── _config.yml
│   └── _toc.yml
├── tests/                         # Pruebas automáticas de la API
│   ├── test_api_local.py
│   └── test_api_quick.py
├── ejemplo_uso_api.py             # Guía rápida para consumir la API
├── requirements.txt               # Dependencias del proyecto
├── Dockerfile                     # Imagen Docker lista para producción
├── .github/workflows/ci.yml       # Pipeline CI/CD en GitHub Actions
├── .gitignore
└── README.md
```

---

## Instalación y Ejecución

### Prerrequisitos

- Python 3.10+
- pip
- Docker (opcional, para containerización)
- Git

### 1. Clonar el Repositorio

```bash
git clone https://github.com/MrSpain2104/Telco-Customer-Churn-Mlops.git
cd Telco-Customer-Churn-Mlops
```

### 2. Crear Entorno Virtual

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar Notebooks

```bash
# Abrir Jupyter Notebook
jupyter notebook

# O usar VS Code con la extensión de Jupyter
```

**Orden de ejecución:**
1. `1_eda_preprocessing.ipynb` → Análisis y limpieza de datos
2. `2_model_training.ipynb` → Entrenamiento de modelos
3. `3_interpretability.ipynb` → Interpretabilidad con LIME

### 5. Ejecutar la API Localmente

```bash
# Desde el directorio raíz
uvicorn app.api:app --reload --port 8000
```

La API estará disponible en:
- **Aplicación**: http://localhost:8000
- **Documentación interactiva**: http://localhost:8000/docs
- **Documentación alternativa**: http://localhost:8000/redoc

---

## Docker

### Construir la Imagen

```bash
docker build -t telco-churn-api .
```

### Ejecutar el Contenedor

```bash
# Crear y ejecutar (primera vez)
docker run -d --name telco-churn-container -p 8000:8000 telco-churn-api

# Detener
docker stop telco-churn-container

# Iniciar (después de creado)
docker start telco-churn-container

# Ver logs
docker logs telco-churn-container

# Eliminar contenedor
docker rm telco-churn-container
```

### Verificar Salud del Contenedor

```bash
curl http://localhost:8000/health
```

---

## Uso de la API

### Endpoints Disponibles

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Información básica de la API |
| GET | `/health` | Estado de salud del servicio |
| POST | `/predict` | Predicción individual de churn |
| POST | `/predict-batch` | Predicción batch (múltiples clientes) |
| GET | `/model-info` | Información del modelo cargado |

### Ejemplo de Predicción

#### Usando curl (PowerShell)

```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{
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
  }'
```

#### Respuesta Esperada

```json
{
  "churn_probability": 0.7854,
  "prediction": "Yes",
  "risk_level": "High",
  "confidence": 0.7854
}
```

#### Usando Python

```python
import requests

url = "http://localhost:8000/predict"

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

response = requests.post(url, json=customer_data)
print(response.json())
```

---

## Tests

### Ejecutar Tests Unitarios

```bash
# Todos los tests
pytest tests/ -v

# Con cobertura
pytest tests/ --cov=app --cov-report=html

# Ver reporte de cobertura
# Abrir htmlcov/index.html en el navegador
```
---

## CI/CD Pipeline

### GitHub Actions

El proyecto incluye un pipeline automatizado que:

1. **Lint**: Verifica calidad del código con Flake8
2. **Test**: Ejecuta tests unitarios con pytest
3. **Build**: Construye imagen Docker
4. **Test Container**: Verifica que el contenedor funcione
5. **Deploy**: Placeholder para despliegue a producción

### Configuración

Para usar el pipeline en tu repositorio:

1. Fork o clona el proyecto
2. Configura secrets en GitHub (si usas Docker Hub):
   - `DOCKER_USERNAME`
   - `DOCKER_PASSWORD`
3. Haz push a la rama `main` o crea un pull request

---

## Documentación

La documentación completa del proyecto está disponible como Jupyter Book en GitHub Pages:

**[Ver Documentación Completa](https://mrspain2104.github.io/Telco-Customer-Churn-Mlops/)**

La documentación incluye:
- Presentación del proyecto y equipo
- Los 3 notebooks completos con visualizaciones interactivas
- Análisis exploratorio de datos
- Entrenamiento y evaluación de modelos
- Interpretabilidad con LIME

### Construir Localmente

```bash
pip install "jupyter-book<2.0"
jupyter-book build jupyter-book
# Abrir jupyter-book/_build/html/index.html en el navegador
```

---

## Tecnologías Utilizadas

### Data Science & ML
- Python 3.10
- pandas, numpy
- scikit-learn
- XGBoost, CatBoost, LightGBM
- LIME

### API & Deployment
- FastAPI
- Uvicorn
- Pydantic
- Docker

### Visualization
- matplotlib
- seaborn

### Testing & CI/CD
- pytest
- Flake8
- GitHub Actions

---

## Autores

**Universidad del Norte - Curso de Machine Learning**

- **Omar Medina**
- **Andrés España**

---

## Contacto

**Repositorio:** [GitHub - Telco Customer Churn MLOps](https://github.com/MrSpain2104/Telco-Customer-Churn-Mlops)

**Equipo:**
- Omar Medina
- Andrés España

---

## Referencias

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [LIME: Local Interpretable Model-Agnostic Explanations](https://github.com/marcotcr/lime)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

**⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub!**

---

<div align="center">
  <p><strong>Universidad del Norte</strong></p>
  <p>Curso de Machine Learning</p>
  <p>2025</p>
</div>
