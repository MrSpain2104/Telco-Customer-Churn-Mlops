# Predicción de Churn en Telecomunicaciones
## Implementación de MLOps con Machine Learning

<div style="text-align: center; padding: 40px 0;">
  <h2> Proyecto Académico</h2>
  <h3>Universidad del Norte</h3>
  <p style="font-size: 1.2em; margin-top: 20px;">Curso: Machine Learning</p>
</div>

---

### Equipo de Desarrollo

**Integrantes:**
- **Omar Medina** - Análisis y Modelado
- **Andrés España** - MLOps e Implementación

---

## Resumen Ejecutivo

Este proyecto implementa un **sistema completo de predicción de churn** para empresas de telecomunicaciones, utilizando técnicas avanzadas de Machine Learning y mejores prácticas de MLOps.

### Objetivos del Proyecto

1. **Predecir el abandono de clientes** (churn) con alta precisión
2. **Identificar factores clave** que influyen en la decisión de los clientes
3. **Implementar un pipeline MLOps completo** desde el desarrollo hasta la producción
4. **Desplegar una API REST** para predicciones en tiempo real

---

## Contexto del Problema

La **tasa de abandono de clientes** (churn) es uno de los principales desafíos en la industria de telecomunicaciones. Retener clientes existentes es significativamente más rentable que adquirir nuevos, con estudios que muestran que:

- Cuesta **5 veces más** adquirir un nuevo cliente que retener uno existente
- Aumentar la retención en un **5%** puede incrementar las ganancias entre **25% y 95%**
- La tasa promedio de churn en telecomunicaciones es del **15-25% anual**

### Dataset: Telco Customer Churn

- **Fuente:** IBM Sample Data Sets (Kaggle)
- **Registros:** ~7,000 clientes
- **Variables:** 20 características demográficas y de servicios
- **Variable objetivo:** Churn (Yes/No)

---

## Tecnologías Utilizadas

### Machine Learning
- **Algoritmos:** Random Forest, XGBoost, CatBoost, LightGBM
- **Optimización:** GridSearchCV con validación cruzada estratificada
- **Interpretabilidad:** LIME (Local Interpretable Model-agnostic Explanations)

### MLOps Stack
- **API:** FastAPI con documentación automática (Swagger/OpenAPI)
- **Containerización:** Docker con imágenes optimizadas
- **CI/CD:** GitHub Actions con testing automático
- **Despliegue:** Preparado para producción con monitoreo

### Herramientas de Desarrollo
- **Python 3.10+** con entornos virtuales
- **Jupyter Notebooks** para análisis exploratorio
- **Scikit-learn** para pipeline de preprocesamiento
- **Pandas & NumPy** para manipulación de datos
- **Matplotlib & Seaborn** para visualizaciones

---

## Metodología

El proyecto sigue una metodología estructurada en 5 fases:

### 1. Análisis Exploratorio de Datos (EDA)
- Comprensión profunda del dataset
- Análisis de distribuciones y correlaciones
- Identificación de patrones de churn
- Visualizaciones profesionales con paleta coherente
- Análisis de Componentes Principales (PCA)

### 2. Entrenamiento de Modelos
- Preprocesamiento con Pipelines de Scikit-learn
- Entrenamiento de 4 algoritmos de clasificación
- Optimización exhaustiva de hiperparámetros
- Evaluación con múltiples métricas (ROC-AUC, F1-Score, etc.)
- Selección del mejor modelo

### 3. Interpretabilidad con LIME
- Explicaciones a nivel individual
- Identificación de factores de riesgo
- Análisis de sensibilidad
- Recomendaciones accionables para el negocio

### 4. Despliegue y API
- API REST con FastAPI
- Validación de datos con Pydantic
- Documentación interactiva automática
- Endpoints para predicción individual y por lotes

### 5. MLOps y CI/CD
- Dockerización de la aplicación
- Pipeline de integración continua
- Tests automatizados
- Preparación para despliegue en producción

---

## 📞 Contacto y Contribuciones

Este proyecto fue desarrollado como parte del curso de Machine Learning en la **Universidad del Norte**.

**Equipo:**
- Omar Medina
- Andrés España

**Repositorio:** [GitHub - Telco Customer Churn MLOps](https://github.com/MrSpain2104/Telco-Customer-Churn-Mlops)

**Licencia:** MIT

---

<div style="text-align: center; padding: 30px 0; border-top: 2px solid #ddd; margin-top: 50px;">
  <p style="font-size: 0.9em; color: #666;">
     Año 2025 | Universidad del Norte | Machine Learning
  </p>
</div>
