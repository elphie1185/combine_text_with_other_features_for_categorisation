import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder
)

import parameters

class FeatureEngineer:
    def __init__(
            self, 
            feature_numeric=parameters.NUMERICAL_FEATURE, 
            feature_categorical=parameters.CATEGORICAL_FEATURE
        ):
        self.num_preprocessor = self.preprocess_numbers()
        self.cat_preprocessor = self.preprocess_categories()
        self.feature_numeric = feature_numeric
        self.feature_categorical = feature_categorical

    def preprocess_numbers(self):
        return make_pipeline(
            SimpleImputer(strategy="median"), 
            StandardScaler()
        )

    def preprocess_categories(self):
        return make_pipeline(
            SimpleImputer(
                strategy="constant", 
                fill_value="otther", 
                missing_values=np.nan
            ), 
            OneHotEncoder(
                handle_unknown="ignore", 
                sparse_output=False
            )
        )
    
    def create_preprocessor(self): 
        transformers = [
            ("num_preprocessor", self.num_preprocessor, self.feature_numeric),
            ("cat_preprocessor", self.cat_preprocessor, self.feature_categorical)
        ]
        return ColumnTransformer(transformers=transformers, remainder="drop")