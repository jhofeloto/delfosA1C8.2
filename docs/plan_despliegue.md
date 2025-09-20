# Plan de despliegue y arquitectura - delfosA1C8

Objetivo: Aplicación de clase mundial para biomarcadores de diabetes que ofrezca predicción individual, procesamiento masivo y APIs consumibles, junto con una landing page explicativa.

## 1. Arquitectura propuesta
- Frontend ligero servido por FastAPI (templates + estáticos)
- API REST con FastAPI (/api/v1)
- Almacenamiento de modelos (.pkl) y métricas (metrics.json)
- Contenedores Docker y orquestación (docker-compose/Kubernetes)
- Observabilidad (logs estructurados, métricas de entrenamiento)

## 2. Endpoints clave
- GET /: estado de servicio
- GET /health: modelos cargados
- GET /ui: interfaz para uso individual/masivo
- GET /landing: landing de fundamento clínico, técnico y arquitectura
- GET /api/v1/models: lista de modelos disponibles
- GET /api/v1/metrics: métricas de entrenamiento
- POST /api/v1/predict: predicción individual (query param model)
- POST /api/v1/predict/batch: carga CSV y predicción masiva

## 3. Flujo de entrenamiento y versionado
1) scripts/generate_synthetic_data.py
2) scripts/train_models.py -> guarda modelos en /models y metrics.json
3) Versionar artefactos por fecha/semver (ejemplo: models/v1/gradient_boosting.pkl)
4) Publicar métricas y changelog de modelos

## 4. Seguridad y cumplimiento
- CORS restringido por dominio en producción
- Autenticación JWT u OAuth2 (por implementar según entorno)
- Validación de payloads, límites de tamaño de CSV
- Auditoría de accesos y peticiones

## 5. Despliegue
### Opción A: Docker Compose
- docker-compose up --build
- Puertos: 8000 (API/UI)

### Opción B: Kubernetes (EKS/GKE/AKS)
- Imagen de Docker publicada en registry (GHCR/ECR/GCR/ACR)
- Manifests: Deployment, Service, HPA, Ingress (TLS)
- ConfigMaps para parámetros y Secrets para credenciales

### Opción C: Cloud Run / App Service
- Build & deploy directo desde repo
- Escalado automático a 0

## 6. Observabilidad
- Logs estructurados (uvicorn, FastAPI)
- Exportar métricas a Prometheus (extensión futura)
- Trazas distribuidas con OpenTelemetry (futuro)

## 7. Roadmap
- Autenticación y rate limiting
- Gestión de versiones de modelos y AB testing
- Panel de métricas y monitoreo
- Explicabilidad (SHAP) y reporting avanzado
- Internacionalización de la UI
