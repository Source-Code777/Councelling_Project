from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

class Transformers:
    def __init__(self):
        self.categorical_columns = ["Branch_short","Domicile","Reservation"]
        self.numerical_columns = ["Closing Rank"]

        self.preprocessor_ccl = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), self.categorical_columns),
                ("num", StandardScaler(), self.numerical_columns)
            ],
            remainder="passthrough"
        )

    def fit(self, X_train, y_train):
        self.preprocessor_ccl.fit(X_train)
        self.label.fit(y_train)
        return self

    def transform(self, X, y=None):
        X_transformed = self.preprocessor_ccl.transform(X)
        if y is not None:
            y_transformed = self.label.transform(y)
            return X_transformed, y_transformed
        return X_transformed

    def fit_transform(self, X, y):
        X_transformed = self.preprocessor_ccl.fit_transform(X)
        y_transformed = self.label.fit_transform(y)
        return X_transformed, y_transformed
