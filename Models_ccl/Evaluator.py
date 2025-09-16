
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
class Best_Model:
    def __init__(self, model_and_params: dict, random_state: int = None,
                 cv: int = 10, scoring: str = 'accuracy', n_jobs: int = -1):
        self.model_and_params = model_and_params
        self.cat_col = ["Branch_short", "Domicile", "Reservation"]
        self.num_col = ["Closing Rank"]
        self.random_state = random_state
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.label = LabelEncoder()

        self.preprocessor_linear = ColumnTransformer(
            transformers=[
                ("ohe", OneHotEncoder(handle_unknown="ignore"), self.cat_col),
                ("std", StandardScaler(), self.num_col),
            ],
            remainder="drop",
        )
        self.preprocessor_tree = ColumnTransformer(
            transformers=[
                ("ohe", OneHotEncoder(handle_unknown="ignore"), self.cat_col),
            ],
            remainder="passthrough",
        )

        # Let's create variables to store our results and other product's
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        self.best_grid = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def pre_selector(self, model):

        if isinstance(model, (LogisticRegression, SVC)):
            return self.preprocessor_linear
        elif isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier)):
            return self.preprocessor_tree
        else:
            return self.preprocessor_tree

    def train_model(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)
        self.y_train = self.label.fit_transform(self.y_train)
        self.y_test = self.label.transform(self.y_test)

        best_score = -1
        for name, (model, params) in self.model_and_params.items():
            print(f"Running GridSearchCV for {name}")
            preprocessor = self.pre_selector(model)
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ])

            grid_search = GridSearchCV(
                pipeline,
                param_grid=params,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs
            )
            grid_search.fit(self.X_train, self.y_train)

            # Saving results...
            self.results[name] = {
                "best_score": grid_search.best_score_,
                "best_params": grid_search.best_params_,
                "best_estimator": grid_search.best_estimator_,
                "grid_search": grid_search,
            }

            # Updating the list if new_best score if it's better than previous model

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                self.best_model_name = name
                self.best_model = grid_search.best_estimator_
                self.best_grid = grid_search

        print(f"Best Model: {self.best_model_name}, with Score: {best_score:.4f}")
        return self.best_model_name, self.best_model

    def predict(self, X: pd.DataFrame = None):
        if self.best_model is None:
            raise ValueError("No best model selected, call train() first!")
        if X is None:
            X = self.X_test
        return self.best_model.predict(X)

    def evaluate(self, X: pd.DataFrame = None, Y: pd.Series = None):
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train_model() first.")
        if X is None or Y is None:
            X, Y = self.X_test, self.y_test

        y_pred = self.best_model.predict(X)

        metrics = {
            "best_model": self.best_model_name,
            "best_params": self.best_grid.best_params_,
            "best_cv_score": self.best_grid.best_score_,
            "accuracy": accuracy_score(Y, y_pred),
            "precision": precision_score(Y, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(Y, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(Y, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(Y, y_pred).tolist(),
            "classification_report": classification_report(Y, y_pred, zero_division=0)
        }

        print("\n================ Evaluation Results ================\n")
        print(f" Best Model:       {metrics['best_model']}\n")
        print(f" Best Params:      {metrics['best_params']}\n")
        print(f" Best CV Score:    {metrics['best_cv_score']:.4f}\n")
        print(f" Accuracy:         {metrics['accuracy']:.4f}\n")
        print(f" Precision:        {metrics['precision']:.4f}\n")
        print(f" Recall:           {metrics['recall']:.4f}\n")
        print(f" F1 Score:         {metrics['f1_score']:.4f}\n")
        print(" Confusion Matrix:")
        print(metrics["confusion_matrix"])
        print("\n Classification Report:\n")
        print(metrics["classification_report"])
        print("===================================================\n")

        return metrics


    def inverse_transform_labels(self, y_encoded):
        return self.label.inverse_transform(y_encoded)
