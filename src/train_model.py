import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_training_data() -> pd.DataFrame:
    """Load and lightly clean the flight dataset for model training."""
    flights = pd.read_csv("data/flights.csv", low_memory=False)

    # Keep only the columns we want to use
    df = flights[
        [
            "MONTH",
            "DAY_OF_WEEK",
            "AIRLINE",
            "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT",
            "DISTANCE",
            "DEPARTURE_DELAY",
        ]
    ].copy()

    # Remove rows where the target is missing
    df = df.dropna(subset=["DEPARTURE_DELAY"])

    # Optional: remove extreme target outliers to make training more stable
    df = df[(df["DEPARTURE_DELAY"] > -20) & (df["DEPARTURE_DELAY"] < 180)]

    # Sample to keep training fast on a student laptop
    if len(df) > 200000:
        df = df.sample(n=200000, random_state=42)

    return df


def build_pipeline() -> Pipeline:
    """Build a preprocessing + model pipeline."""
    categorical_features = [
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
    ]
    numeric_features = [
        "MONTH",
        "DAY_OF_WEEK",
        "DISTANCE",
    ]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def main() -> None:
    df = load_training_data()

    X = df.drop(columns=["DEPARTURE_DELAY"])
    y = df["DEPARTURE_DELAY"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Model training complete.")
    print(f"Rows used: {len(df):,}")
    print(f"Mean Absolute Error: {mae:.2f} minutes")
    print(f"R^2 Score: {r2:.3f}")

    joblib.dump(pipeline, "models/delay_predictor.joblib")
    print("Saved model to models/delay_predictor.joblib")


if __name__ == "__main__":
    main()