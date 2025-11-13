# Guía de Inicio Rápido - Telco Churn Prediction MLOps

## 📋 Pasos para Ejecutar el Proyecto Completo

### Paso 1: Configuración Inicial del Entorno

```powershell
# 1. Abrir PowerShell en el directorio del proyecto
cd "c:\Users\andre\OneDrive\Escritorio\Machine\Churm"

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno virtual
.\venv\Scripts\Activate.ps1

# 4. Actualizar pip
python -m pip install --upgrade pip

# 5. Instalar dependencias
pip install -r requirements.txt
```

### Paso 2: Ejecutar los Notebooks (EN ORDEN)

#### 2.1. Notebook 1: EDA y Preprocesamiento

```powershell
# Abrir VS Code
code .

# O abrir Jupyter Notebook
jupyter notebook
```

**Ejecutar:** `notebooks/1_eda_preprocessing.ipynb`

**Qué hace:**
- Carga el dataset `data/telco_churn.csv`
- Análisis exploratorio completo
- Limpieza de datos y tratamiento de valores faltantes
- Genera: `data/telco_churn_clean.csv`

**Tiempo estimado:** 5-10 minutos

---

#### 2.2. Notebook 2: Entrenamiento de Modelos

**Ejecutar:** `notebooks/2_model_training.ipynb`

**Qué hace:**
- Carga datos limpios
- Entrena 4 modelos: Random Forest, XGBoost, CatBoost, LightGBM
- Optimiza hiperparámetros con GridSearchCV
- Evalúa con métricas completas
- Genera:
  - `app/model.joblib` (mejor modelo)
  - `app/model_*.joblib` (todos los modelos)
  - `app/model_metrics.csv`
  - `app/feature_importance_*.csv`

**⚠️ IMPORTANTE:** Este notebook tarda ~30-60 minutos dependiendo de tu hardware.

**Tiempo estimado:** 30-60 minutos

---

#### 2.3. Notebook 3: Interpretabilidad con LIME

**Ejecutar:** `notebooks/3_interpretability.ipynb`

**Qué hace:**
- Carga el mejor modelo
- Explica predicciones individuales con LIME
- Análisis de casos representativos
- Análisis de sensibilidad

**Tiempo estimado:** 10-15 minutos

---

### Paso 3: Ejecutar la API Localmente

```powershell
# Desde el directorio raíz
uvicorn app.api:app --reload --port 8001
```

**Verificar:**
- Aplicación: http://localhost:8001
- Documentación: http://localhost:8001/docs
- Salud: http://localhost:8001/health

**Prueba rápida:**

```powershell
# En otra terminal (con el entorno virtual activado)
python test_api_local.py
```

---

### Paso 4: Ejecutar Tests Unitarios

```powershell
# Tests de la API
pytest tests/test_api.py -v

# Tests del modelo
pytest tests/test_model.py -v

# Todos los tests con cobertura
pytest tests/ --cov=app --cov-report=html

# Ver reporte de cobertura
# Abrir: htmlcov/index.html en el navegador
```

---

### Paso 5: Docker (Opcional)

#### 5.1. Construir la Imagen Docker

```powershell
docker build -t telco-churn-api .
```

#### 5.2. Ejecutar el Contenedor

```powershell
# Crear y ejecutar
docker run -d --name telco-churn-container -p 8000:8000 telco-churn-api

# Ver logs
docker logs telco-churn-container

# Verificar salud
curl http://localhost:8000/health
```

#### 5.3. Prueba desde Docker

```powershell
# Ejecutar el script de pruebas (apuntando a puerto 8000)
# Editar test_api_local.py: BASE_URL = "http://localhost:8000"
python test_api_local.py
```

#### 5.4. Gestión del Contenedor

```powershell
# Detener
docker stop telco-churn-container

# Iniciar (después de creado)
docker start telco-churn-container

# Ver estado
docker ps -a | Select-String "telco-churn"

# Eliminar
docker rm telco-churn-container
```

---

### Paso 6: Pruebas Manuales de la API

#### Opción 1: Usar la Documentación Interactiva

1. Ir a http://localhost:8000/docs (o 8001 si usas local)
2. Expandir `/predict`
3. Hacer clic en "Try it out"
4. Modificar el JSON de ejemplo
5. Hacer clic en "Execute"

#### Opción 2: Usar curl (PowerShell)

```powershell
# Ejemplo 1: Cliente con alto riesgo de churn
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

# Ejemplo 2: Cliente con bajo riesgo de churn
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{
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
  }'
```

#### Opción 3: Usar Python (requests)

```python
import requests

url = "http://localhost:8000/predict"

customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    # ... resto de los campos
}

response = requests.post(url, json=customer)
print(response.json())
```

---

## 🔧 Solución de Problemas Comunes

### Problema 1: "El modelo no está cargado"

**Causa:** No se ejecutó el Notebook 2 o el archivo model.joblib no existe.

**Solución:**
```powershell
# Verificar que existe el modelo
ls app/model.joblib

# Si no existe, ejecutar el Notebook 2 completo
```

### Problema 2: Error al instalar dependencias

**Causa:** Conflictos de versiones o problemas de compilación.

**Solución:**
```powershell
# Reinstalar con flag de actualización
pip install --upgrade --force-reinstall -r requirements.txt

# O instalar dependencias problemáticas individualmente
pip install xgboost
pip install catboost
pip install lightgbm
```

### Problema 3: Puerto ya en uso

**Causa:** Otro proceso está usando el puerto 8000.

**Solución:**
```powershell
# Usar otro puerto
uvicorn app.api:app --reload --port 8001

# O detener el proceso que usa el puerto 8000
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process
```

### Problema 4: Docker no inicia el contenedor

**Causa:** Modelo no copiado al contenedor o error en Dockerfile.

**Solución:**
```powershell
# Verificar logs del contenedor
docker logs telco-churn-container

# Reconstruir sin caché
docker build --no-cache -t telco-churn-api .
```

### Problema 5: Tests fallan

**Causa:** Modelo no cargado o cambios en la estructura.

**Solución:**
```powershell
# Verificar que el modelo existe
ls app/model.joblib

# Ejecutar tests con más información
pytest tests/ -v -s
```

---

## 📊 Verificación de que Todo Funciona

### Checklist Completo

- [ ] Entorno virtual creado y activado
- [ ] Dependencias instaladas sin errores
- [ ] Notebook 1 ejecutado → `data/telco_churn_clean.csv` creado
- [ ] Notebook 2 ejecutado → `app/model.joblib` creado
- [ ] Notebook 3 ejecutado → Explicaciones LIME generadas
- [ ] API local funciona → http://localhost:8001/docs accesible
- [ ] Test local exitoso → `python test_api_local.py` pasa
- [ ] Tests unitarios pasan → `pytest tests/ -v` exitoso
- [ ] Docker construye → `docker build` exitoso
- [ ] Docker ejecuta → Contenedor corriendo y respondiendo
- [ ] Predicciones funcionan → curl/Postman retorna predicciones válidas

---

## 🎯 Flujo de Trabajo Recomendado

### Para Desarrollo

```powershell
# 1. Activar entorno
.\venv\Scripts\Activate.ps1

# 2. Ejecutar API en modo desarrollo
uvicorn app.api:app --reload --port 8001

# 3. En otra terminal: ejecutar tests automáticamente
pytest tests/ --cov=app -v --watch
```

### Para Producción (Simulada)

```powershell
# 1. Construir imagen Docker
docker build -t telco-churn-api:v1.0 .

# 2. Ejecutar contenedor
docker run -d --name telco-api-prod -p 80:8000 telco-churn-api:v1.0

# 3. Verificar salud
curl http://localhost/health

# 4. Pruebas de carga (opcional, requiere herramientas adicionales)
# locust -f load_test.py --host=http://localhost
```

---

## 📚 Recursos Adicionales

### Documentación

- FastAPI Docs: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc
- OpenAPI Schema: http://localhost:8001/openapi.json

### Logs y Monitoreo

```powershell
# Ver logs de la API (local)
# Los logs aparecen en la terminal donde ejecutaste uvicorn

# Ver logs de Docker
docker logs -f telco-churn-container

# Ver logs de tests
pytest tests/ -v -s --log-cli-level=INFO
```

---

## 🚀 Próximos Pasos Sugeridos

1. **Explorar la documentación interactiva** en /docs
2. **Modificar hiperparámetros** en el Notebook 2 para mejorar el modelo
3. **Agregar nuevas características** al modelo
4. **Implementar autenticación** en la API (JWT)
5. **Agregar monitoreo** con Prometheus/Grafana
6. **Desplegar en la nube** (AWS, GCP, Azure)
7. **Implementar CI/CD completo** en GitHub Actions

---

## 💡 Tips y Mejores Prácticas

1. **Siempre activar el entorno virtual** antes de trabajar
2. **Ejecutar tests antes de hacer commits**
3. **Usar git para versionar el código** (no el modelo si es muy grande)
4. **Documentar cambios en los notebooks**
5. **Mantener requirements.txt actualizado**
6. **Usar .gitignore** para excluir archivos innecesarios
7. **Revisar logs regularmente** para detectar errores

---

## 📞 Soporte

Si encuentras problemas:

1. Revisa esta guía primero
2. Consulta el README.md principal
3. Revisa los logs de error
4. Busca en la documentación de FastAPI/scikit-learn
5. Crea un issue en el repositorio (si es un proyecto público)

---

**¡Feliz coding! 🎉**
