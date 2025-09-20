# 🏥 delfosA1C8 - Sistema Predictivo de Diabetes Mellitus Tipo 2

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)

Aplicación de clase mundial para biomarcadores de riesgo de Diabetes Tipo 2: predicción individual y masiva de glucosa en ayunas, categorización clínica y API lista para integrar. Incluye landing page, UI de demo, entrenamiento de modelos y plan de despliegue.

## 🔎 Índice
- Características clave
- Estructura del proyecto
- Instalación y ejecución
- Entrenamiento de modelos
- API (endpoints)
- UI y Landing
- Despliegue (Docker/Supervisor)
- Métricas y resultados
- Roadmap

---

## ✨ Características clave
- Predicción individual (JSON) y masiva (CSV) con selección de modelo.
- Modelos incluidos: Linear, Ridge, Random Forest, Gradient Boosting.
- Métricas en vivo expuestas vía `/api/v1/metrics` y mostradas en la landing (Chart.js).
- UI de demo para predicciones y carga de CSV.
- Landing page con fundamento clínico/técnico y arquitectura.
- Entrenamiento reproducible y exportación de artefactos: modelos `.pkl`, `metrics.json`, `scaler.pkl`, `feature_order.json`.
- Contenedores (Dockerfile, docker-compose) y servicio con Supervisor en producción simple.

## 🗂️ Estructura del proyecto
```
.
├── src
│   ├── api/app.py           # FastAPI (endpoints, UI, landing, estáticos)
│   ├── data/preprocessor.py # Pipeline de datos (ingeniería de variables y escalado)
│   ├── models/              # Wrapper(s) de modelos
│   └── web/
│       ├── templates/
│       │   ├── index.html   # Demo UI (individual + batch)
│       │   └── landing.html # Landing con métricas en vivo
│       └── static/
│           ├── styles.css
│           └── sample_batch.csv # CSV de ejemplo para cargas masivas
├── scripts
│   ├── generate_synthetic_data.py # Generación de datos sintéticos
│   └── train_models.py            # Entrenamiento y guardado de artefactos
├── models/                 # Artefactos de modelos (.pkl, metrics.json, scaler, feature_order)
├── docs/
│   └── plan_despliegue.md  # Plan y arquitectura de despliegue
├── notebooks/
│   └── analisis_comparativo_diabetes.py # Análisis y visualizaciones
├── Dockerfile
├── docker-compose.yml
├── supervisord.conf
├── requirements.txt
└── README.md
```

## ⚙️ Instalación y ejecución (local)
```bash
# 1) Instalar dependencias
pip install -r requirements.txt

# 2) Entrenar y generar artefactos (modelos + métricas + scaler + orden de features)
python scripts/train_models.py

# 3) Ejecutar API (desarrollo)
uvicorn src.api.app:app --reload
# Documentación interactiva en: http://localhost:8000/docs
```

## 🧠 Entrenamiento de modelos
- Datos de ejemplo: se generan sintéticamente con `generate_synthetic_data.py` (también desde `train_models.py` si no existen).
- Modelos entrenados: Linear, Ridge, RandomForest, GradientBoosting.
- Artefactos guardados en `models/`:
  - `linear_regression.pkl`, `ridge_regression.pkl`, `random_forest.pkl`, `gradient_boosting.pkl`
  - `metrics.json` (R², RMSE, MAE por modelo)
  - `scaler.pkl` (StandardScaler para inferencia)
  - `feature_order.json` (orden de columnas usado en entrenamiento)

## 🌐 API (endpoints principales)
- GET `/` → Estado del servicio
- GET `/health` → Modelos cargados
- GET `/api/v1/models` → Lista de modelos y orden de features
- GET `/api/v1/metrics` → Métricas del último entrenamiento
- POST `/api/v1/predict` → Predicción individual
  - Query param: `model` (opcional; default: gradient_boosting)
  - Body JSON: `edad, talla, peso, imc, tas, tad, perimetro_abdominal, puntaje_total, riesgo_dm`
- POST `/api/v1/predict/batch` → Predicción masiva (CSV)
  - Query param: `model`
  - Form-Data: `file=@archivo.csv`
  - Cabeceras requeridas: `edad,talla,peso,imc,tas,tad,perimetro_abdominal,puntaje_total,riesgo_dm`

Respuesta (individual) ejemplo:
```json
{
  "modelo": "random_forest",
  "glucosa_predicha": 118.11,
  "categoria": "prediabetes",
  "confianza": 0.87
}
```

## 🖥️ UI y Landing
- Demo UI: `/ui` — formulario para predicción individual y carga CSV.
- Landing: `/landing` — mensaje de valor, métricas en vivo, arquitectura, cómo usar la API y descarga de CSV de ejemplo.
- CSV de ejemplo: `/static/sample_batch.csv`.

## 🚀 Despliegue
- Docker Compose (desarrollo/QA):
```bash
docker compose up --build
```
- Supervisor (producción simple en VM/host):
```bash
pip install supervisor
supervisord -c supervisord.conf
supervisorctl -c supervisord.conf status
```
- Kubernetes/Cloud: ver docs/plan_despliegue.md (arquitectura, TLS, HPA, CI/CD).

## 📈 Métricas y resultados (ejemplo sintético)
- Se reportan métricas por modelo en `models/metrics.json` y vía `/api/v1/metrics`.
- La landing consume estas métricas y grafica R² y RMSE con Chart.js.
- Nota: con datos reales, re-entrenar y documentar resultados clínicos/estadísticos.

## 🗺️ Roadmap
- Autenticación (OAuth2/JWT) y rate limiting.
- Monitorización (Prometheus/Grafana) y detección de drift.
- Explicabilidad (SHAP), reporting clínico y validación prospectiva.
- CI/CD con GitHub Actions y despliegue a Kubernetes/Cloud Run.

## 📄 Licencia
MIT License — ver [LICENSE](LICENSE)
