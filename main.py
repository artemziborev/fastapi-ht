from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import datetime
from typing import List, Optional
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Инициализация FastAPI
app = FastAPI(
    title="ML Model Serving API",
    description="API для обслуживания моделей машинного обучения",
    version="1.0.0"
)

# Конфигурация путей
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "iris_model.pkl")

# Создаем директорию и модель, если их нет
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    # Загрузка данных и обучение модели
    iris = load_iris()
    X, y = iris.data, iris.target
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
else:
    model = joblib.load(MODEL_PATH)


# Pydantic модели для валидации
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionInput(BaseModel):
    features: List[float]
    model_version: Optional[str] = "v1"


class PredictionOutput(BaseModel):
    prediction: int
    probabilities: List[float]
    model_version: str
    timestamp: str
    status: str = "success"


# Endpoints
@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Предсказание класса цветка ириса по параметрам

    Параметры:
    - sepal_length: длина чашелистика (см)
    - sepal_width: ширина чашелистика (см)
    - petal_length: длина лепестка (см)
    - petal_width: ширина лепестка (см)
    """
    try:
        features = np.array(input_data.features).reshape(1, -1)

        if features.shape[1] != 4:
            raise ValueError("Требуется ровно 4 признака")

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0].tolist()

        return {
            "prediction": int(prediction),
            "probabilities": probabilities,
            "model_version": input_data.model_version,
            "timestamp": datetime.datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка предсказания: {str(e)}"
        )


@app.get("/model-info")
async def get_model_info():
    """Информация о загруженной модели"""
    return {
        "model_type": "RandomForestClassifier",
        "model_version": "v1",
        "input_features": [
            "sepal_length (cm)",
            "sepal_width (cm)",
            "petal_length (cm)",
            "petal_width (cm)"
        ],
        "classes": ["setosa", "versicolor", "virginica"],
        "n_estimators": model.n_estimators,
        "accuracy": 0.96  # Примерное значение для демонстрации
    }


@app.get("/health")
async def health_check():
    """Проверка работоспособности API"""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)