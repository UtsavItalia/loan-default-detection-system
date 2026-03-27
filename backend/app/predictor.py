import numpy as np
import joblib
import json
import os


class DefaultPredictor:
    def __init__(self, model_dir: str = "models"):

        print("Loading models artifacts...")
        model_path = os.path.join(model_dir, "best_model.joblib")
        self.model = joblib.load(model_path)
        print(f"  Model loaded from {model_path}")

        scaler_path = os.path.join(model_dir, "scaler.joblib")
        self.scaler = joblib.load(scaler_path)
        print(f"  Scaler loaded from {scaler_path}")

        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        print(f"  Metadata loaded from {metadata_path}")

        self.feature_columns = self.metadata["feature_columns"]
        print("All artifacts loaded successfully.")

    def predict(self, features: dict) -> dict:
        input_data = np.array([[features[col] for col in self.feature_columns]])
        print(f"\n{input_data}")
        input_scaled = self.scaler.transform(input_data)
        print(f"\n {input_scaled}")

        prediction = int(self.model.predict(input_scaled)[0])
        probability = float(self.model.predict_proba(input_scaled)[0][1])

        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk_level": risk_level,
        }

    def get_model_stats(self) -> dict:
        return {"model_results": self.metadata["model_results"]}

    def get_feature_importance(self) -> dict:
        return {"feature_importance": self.metadata["feature_importance"]}
