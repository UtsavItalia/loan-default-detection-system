from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelStatsResponse,
    FeatureImportanceResponse,
)
from app.predictor import DefaultPredictor

app = FastAPI(
    title="Loan Default Detection API",
    description="Predict loan default risk using ML models with SHAP explainability",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = DefaultPredictor()


@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Loan Default Detection API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    print(request)
    features = request.model_dump()
    print(f"\n{features}")
    result = predictor.predict(features)
    return result


@app.get("/model-stats", response_model=ModelStatsResponse)
def model_stats():
    return predictor.get_model_stats()


@app.get("/feature-importance", response_model=FeatureImportanceResponse)
def feature_importance():
    return predictor.get_feature_importance()
