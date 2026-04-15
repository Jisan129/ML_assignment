import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MODELS_DIR = Path(__file__).parent.parent / "models"


def train(model, X, y, test_size=0.2, random_state=42):
    """Split data, fit model, and return results."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    return model, X_test, y_test, y_pred


def save_model(model, name: str) -> None:
    """Persist a trained model to models/."""
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    print(f"Model saved to models/{name}.joblib")


def load_model(name: str):
    """Load a persisted model from models/."""
    return joblib.load(MODELS_DIR / f"{name}.joblib")
