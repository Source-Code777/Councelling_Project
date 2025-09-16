import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

from Pre_Processing import Transformers


class Logistic_Model_class:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.transformer = Transformers()
        self.model = LogisticRegression(random_state=self.random_state, max_iter=10000)
        self.label = LabelEncoder()
        self.pipeline = Pipeline(steps=[
            ("preprocessor", self.transformer.preprocessor_ccl),
            ("log_reg", self.model)
        ])

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
                raise ValueError("No test data available. Pass X manually or train the model first.")
            X = self.X_test
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame = None):
        if X is None:
            if self.X_test is None:
                raise ValueError("No test data available. Pass X manually or train the model first.")
            X = self.X_test
        return self.pipeline.predict_proba(X)

    def evaluate(self):
        y_pred = self.pipeline.predict(self.X_test)
        metrics = {
            "accuracy": accuracy_score(self.Y_test, y_pred),
            "precision": precision_score(self.Y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(self.Y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(self.Y_test, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(self.Y_test, y_pred).tolist(),
            "classification_report": classification_report(self.Y_test, y_pred, zero_division=0)
        }
        for key, value in metrics.items():
            print(f"{key}:")
            print(value)
            print()
