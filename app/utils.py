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