import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


class RandomForest_Model:
    def __init__(self, categorical_columns=None, numerical_columns=None, random_state: int = 42, **kwargs):
        self.random_state = random_state
        self.categorical_columns = categorical_columns or ["Branch_short", "Domicile", "Reservation"]
        self.numerical_columns = numerical_columns or ["Closing Rank"]
        self.label = LabelEncoder()
        self.model = RandomForestClassifier(random_state=self.random_state, **kwargs)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), self.categorical_columns)
            ],
            remainder="passthrough"
        )

        self.pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("random_forest", self.model)
        ])

        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

    def train(self, X: pd.DataFrame, Y: pd.Series, test_size: float = 0.2):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=Y
        )
        self.Y_train = self.label.fit_transform(self.Y_train)
        self.Y_test = self.label.transform(self.Y_test)
        self.pipeline.fit(self.X_train, self.Y_train)

    def predict(self, X: pd.DataFrame = None):
        if X is None:
            if self.X_test is None:
                raise ValueError("No test data available. Train the model first or pass X manually.")
            X = self.X_test
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame = None):
        if X is None:
            if self.X_test is None:
                raise ValueError("No test data available. Train the model first or pass X manually.")
            X = self.X_test
        return self.pipeline.predict_proba(X)

    def evaluate(self, X: pd.DataFrame = None, Y: pd.Series = None):
        if X is None or Y is None:
            if self.X_test is None or self.Y_test is None:
                raise ValueError("No test data available. Train the model first or pass X and Y manually.")
            X, Y = self.X_test, self.Y_test

        y_pred = self.pipeline.predict(X)
        metrics = {
            "accuracy": accuracy_score(Y, y_pred),
            "precision": precision_score(Y, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(Y, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(Y, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(Y, y_pred).tolist(),
            "classification_report": classification_report(Y, y_pred, zero_division=0)
        }
        return metrics

    def print_metrics(self, metrics: dict):
        for key, value in metrics.items():
            print(f"{key}:")
            print(value)
            print()

    def inverse_transform_labels(self, y_encoded):
        return self.label.inverse_transform(y_encoded)
