"""
FastAPI Application
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import io
import json
import numpy as np
import joblib
from pathlib import Path

app = FastAPI(title="delfosA1C8 API", version="1.1.0", description="API para predicción de glucosa y clasificación de estado glucémico. Incluye predicción individual, masiva y selección de modelo.")

# Static files and templates for web UI
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
templates = Jinja2Templates(directory="src/web/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {}
DEFAULT_MODEL = "gradient_boosting"

MODEL_FILE_MAP = {
    "linear_regression": "models/linear_regression.pkl",
    "ridge_regression": "models/ridge_regression.pkl",
    "random_forest": "models/random_forest.pkl",
    "gradient_boosting": "models/gradient_boosting.pkl",
}

@app.on_event("startup")
async def load_model():
    # Cargar todos los modelos disponibles que existan en disco
    for key, path in MODEL_FILE_MAP.items():
        p = Path(path)
        if p.exists():
            try:
                models[key] = joblib.load(p)
            except Exception:
                pass

class PredictionRequest(BaseModel):
    edad: float = 55
    imc: float = 28.5
    tas: float = 135
    tad: float = 85
    perimetro_abdominal: float = 95
    peso: float = 70
    talla: float = 165
    riesgo_dm: float = 0.5
    puntaje_total: float = 10

class PredictionResponse(BaseModel):
    modelo: str
    glucosa_predicha: float
    categoria: str
    confianza: float

@app.get("/")
def root():
    return {"status": "active", "project": "delfosA1C8", "api_version": app.version}

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": list(models.keys())}

class BatchPredictionResponse(BaseModel):
    modelo: str
    n_registros: int
    resultados: List[PredictionResponse]

@app.post("/api/v1/predict")
def predict(request: PredictionRequest, model: str | None = Query(default=None, description="Modelo a utilizar")):
    # Selección de modelo
    model_key = (model if model else DEFAULT_MODEL)
    if model_key not in models:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_key}' no disponible. Disponibles: {list(models.keys())}")

    features = np.array([[
        request.edad, request.talla, request.peso, request.imc,
        request.tas, request.tad, request.perimetro_abdominal,
        request.puntaje_total, request.riesgo_dm
    ]])

    estimator = models[model_key]
    glucose = float(estimator.predict(features)[0])

    if glucose < 100:
        category = "normal"
    elif glucose <= 126:
        category = "prediabetes"
    else:
        category = "diabetes"

    return PredictionResponse(
        modelo=model_key,
        glucosa_predicha=round(glucose, 2),
        categoria=category,
        confianza=0.87
    )

@app.post("/api/v1/predict/batch")
def predict_batch(file: UploadFile = File(...), model: str | None = Query(default=None)) -> BatchPredictionResponse:
    if model is None:
        model = DEFAULT_MODEL
    if model not in models:
        raise HTTPException(status_code=404, detail=f"Modelo '{model}' no disponible. Disponibles: {list(models.keys())}")

    try:
        content = file.file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo CSV: {e}")

    required_cols = [
        'edad', 'talla', 'peso', 'imc', 'tas', 'tad',
        'perimetro_abdominal', 'puntaje_total', 'riesgo_dm'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")

    features = df[required_cols].to_numpy()
    estimator = models[model]
    preds = estimator.predict(features)

    resultados: List[PredictionResponse] = []
    for glucose in preds:
        glucose = float(glucose)
        if glucose < 100:
            category = "normal"
        elif glucose <= 126:
            category = "prediabetes"
        else:
            category = "diabetes"
        resultados.append(PredictionResponse(
            modelo=model,
            glucosa_predicha=round(glucose, 2),
            categoria=category,
            confianza=0.87
        ))

    return BatchPredictionResponse(modelo=model, n_registros=len(resultados), resultados=resultados)

@app.get("/api/v1/models")
def list_models():
    return {"modelos_disponibles": list(models.keys()), "default": DEFAULT_MODEL}

@app.get("/api/v1/metrics")
def get_metrics():
    metrics_path = Path("models/metrics.json")
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="No hay métricas disponibles aún. Entrene los modelos primero.")
    with open(metrics_path, "r") as f:
        data = json.load(f)
    return data

# Web UI routes
@app.get("/ui", response_class=HTMLResponse)
def ui_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/landing", response_class=HTMLResponse)
def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})
