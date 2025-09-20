"""
FastAPI Application
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sklearn.preprocessing import StandardScaler
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
scaler: StandardScaler | None = None
feature_order: list[str] | None = None
DEFAULT_MODEL = "gradient_boosting"

MODEL_FILE_MAP = {
    "linear_regression": "models/linear_regression.pkl",
    "ridge_regression": "models/ridge_regression.pkl",
    "random_forest": "models/random_forest.pkl",
    "gradient_boosting": "models/gradient_boosting.pkl",
}

@app.on_event("startup")
async def load_model():
    global scaler, feature_order
    # Cargar todos los modelos disponibles que existan en disco
    for key, path in MODEL_FILE_MAP.items():
        p = Path(path)
        if p.exists():
            try:
                models[key] = joblib.load(p)
            except Exception:
                pass
    # Cargar scaler y orden de features si existen
    sp = Path("models/scaler.pkl")
    if sp.exists():
        try:
            scaler = joblib.load(sp)
        except Exception:
            scaler = None
    fop = Path("models/feature_order.json")
    if fop.exists():
        try:
            obj = json.load(open(fop))
            feature_order = obj.get("features")
        except Exception:
            feature_order = None

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

    # Construir vector de características consistente con entrenamiento
    input_map = {
        'edad': request.edad,
        'talla': request.talla,
        'peso': request.peso,
        'imc': request.imc,
        'tas': request.tas,
        'tad': request.tad,
        'perimetro_abdominal': request.perimetro_abdominal,
        'puntaje_total': request.puntaje_total,
        'riesgo_dm': request.riesgo_dm,
        # Features derivadas si fueron usadas durante entrenamiento
        'presion_arterial_media': (request.tas + 2*request.tad)/3,
        'ratio_cintura_altura': request.perimetro_abdominal / request.talla if request.talla else 0,
        'indice_masa_corporal_cat': int(pd.cut([request.imc], bins=[0,18.5,25,30,100], labels=[0,1,2,3])[0]) if request.imc is not None else 0,
    }
    # Selección y orden de columnas
    cols = feature_order if feature_order else list(input_map.keys())
    row = [input_map.get(c, 0) for c in cols]
    X = np.array([row], dtype=float)
    if scaler is not None and feature_order is not None:
        X = scaler.transform(X)

    estimator = models[model_key]
    glucose = float(estimator.predict(X)[0])

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

    # Añadir derivadas y ordenar columnas
    df = df.copy()
    df['presion_arterial_media'] = (df['tas'] + 2*df['tad'])/3
    df['ratio_cintura_altura'] = df['perimetro_abdominal'] / df['talla']
    # IMC categórico
    df['indice_masa_corporal_cat'] = pd.cut(
        df['imc'], bins=[0,18.5,25,30,100], labels=[0,1,2,3]
    ).astype('int64')

    cols = feature_order if feature_order else required_cols
    # Asegurar todas las columnas
    for c in (feature_order or []):
        if c not in df.columns:
            df[c] = 0
    X = df[cols].to_numpy(dtype=float)
    if scaler is not None and feature_order is not None:
        X = scaler.transform(X)

    estimator = models[model]
    preds = estimator.predict(X)

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
    return {"modelos_disponibles": list(models.keys()), "default": DEFAULT_MODEL, "feature_order": feature_order}

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
