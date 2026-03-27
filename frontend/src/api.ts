import axios from "axios";

const API = axios.create({
    baseURL: 'http://localhost:8000'
})

export interface PredictionRequest {
    EXT_SOURCE_1: number;
    EXT_SOURCE_2: number;
    EXT_SOURCE_3: number;
    DAYS_BIRTH: number;
    DAYS_EMPLOYED: number;
    DAYS_REGISTRATION: number;
    DAYS_ID_PUBLISH: number;
    DAYS_LAST_PHONE_CHANGE: number;
    REGION_RATING_CLIENT: number;
    REGION_RATING_CLIENT_W_CITY: number;
    REG_CITY_NOT_WORK_CITY: number;
    FLAG_EMP_PHONE: number;
    FLAG_DOCUMENT_3: number;
    AMT_CREDIT: number;
    AMT_GOODS_PRICE: number;
}

export interface PredictionResponse {
    prediction: number;
    probability: number;
    risk_level: string;
}

export interface ModelResult {
    precision: number;
    recall: number;
    f1_score: number;
    roc_auc: number;
    confusion_matrix: number[][];
}

export interface ModelStatsResponse {
    model_results: Record<string, ModelResult>;
}

export interface FeatureImportanceResponse {
    feature_importance: Record<string, number>;
}

export const getPrediction = async (data: PredictionRequest): Promise<PredictionResponse> => {
    const response = await API.post<PredictionResponse>('/predict', data);
    return response.data;
};

export const getModelStats = async (): Promise<ModelStatsResponse> => {
    const response = await API.get<ModelStatsResponse>('/model-stats');
    return response.data;
};

export const getFeatureImportance = async (): Promise<FeatureImportanceResponse> => {
    const response = await API.get<FeatureImportanceResponse>('/feature-importance');
    return response.data;
};
