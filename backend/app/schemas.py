from pydantic import BaseModel


class PredictionRequest(BaseModel):

    EXT_SOURCE_1: float = 0.5
    EXT_SOURCE_2: float = 0.5
    EXT_SOURCE_3: float = 0.5
    DAYS_BIRTH: float = -10000
    DAYS_EMPLOYED: float = -2000
    DAYS_REGISTRATION: float = -5000
    DAYS_ID_PUBLISH: float = -3000
    DAYS_LAST_PHONE_CHANGE: float = -500
    REGION_RATING_CLIENT: int = 2
    REGION_RATING_CLIENT_W_CITY: int = 2
    REG_CITY_NOT_WORK_CITY: int = 0
    FLAG_EMP_PHONE: int = 1
    FLAG_DOCUMENT_3: int = 1
    AMT_CREDIT: float = 500000
    AMT_GOODS_PRICE: float = 450000


class PredictionResponse(BaseModel):

    prediction: int
    probability: float
    risk_level: str


class ModelStatsResponse(BaseModel):

    model_results: dict


class FeatureImportanceResponse(BaseModel):

    feature_importance: dict
