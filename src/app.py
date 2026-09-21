from pathlib import Path
from pickle import dump

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "housing.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FEATURES = ["MedInc", "Latitude", "Longitude"]


def load_features(path: Path = RAW_PATH) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data[FEATURES].copy()


def split_data(X: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
    return X_train.copy(), X_test.copy()


def train_kmeans(X_train: pd.DataFrame, n_clusters: int = 6, random_state: int = 42) -> KMeans:
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    model.fit(X_train[FEATURES])
    return model


def assign_clusters(model: KMeans, X: pd.DataFrame) -> pd.DataFrame:
    labeled = X.copy()
    labeled["cluster"] = model.predict(X[FEATURES])
    labeled["cluster"] = labeled["cluster"].astype("category")
    return labeled


def train_classifier(X_train: pd.DataFrame, random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, max_depth=14, random_state=random_state, n_jobs=-1)
    model.fit(X_train[FEATURES], X_train["cluster"])
    return model


def save_artifacts(
    kmeans: KMeans,
    classifier: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "housing_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "housing_test.csv", index=False)
    with open(MODELS_DIR / "kmeans_housing.sav", "wb") as file:
        dump(kmeans, file)
    with open(MODELS_DIR / "random_forest_housing.sav", "wb") as file:
        dump(classifier, file)


def main() -> None:
    X = load_features()
    X_train, X_test = split_data(X)

    kmeans = train_kmeans(X_train)
    X_train = assign_clusters(kmeans, X_train)
    X_test = assign_clusters(kmeans, X_test)

    classifier = train_classifier(X_train)
    y_pred = classifier.predict(X_test[FEATURES])

    save_artifacts(kmeans, classifier, X_train, X_test)

    print(f"Train houses: {len(X_train)}")
    print(f"Test houses: {len(X_test)}")
    print("Train cluster counts:")
    print(X_train["cluster"].value_counts().sort_index().to_string())
    print("Test cluster counts:")
    print(X_test["cluster"].value_counts().sort_index().to_string())
    print(f"\nClassifier test accuracy: {accuracy_score(X_test['cluster'], y_pred):.4f}")
    print(classification_report(X_test["cluster"], y_pred, digits=3))
    for saved in sorted(MODELS_DIR.glob("*.sav")):
        print(f"Saved {saved.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
