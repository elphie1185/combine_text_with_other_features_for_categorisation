import numpy as np
import pandas as pd
import re

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src import feature_engineer, light_gbm_model, parameters, text_tokenizer

# load the data and create rating names category
wine_df = pd.read_csv("data/wine_data.csv")
bins = [0, 87, 94, np.inf]
names = ["neutral", "good", "excellent"]
wine_df["rating"] = pd.cut(wine_df["points"], bins, labels=names)
wine_df = wine_df[parameters.FEATURES + [parameters.TARGET]]

train_df, test_df = train_test_split(wine_df, test_size=0.2, random_state=42)

# create the column transformer
feature_engineer = feature_engineer.FeatureEngineer()
feature_engineer.create_preprocessor().set_output(transform="pandas")
preprocessed_num_cat_features_df = feature_engineer.fit_transform(train_df[[
        parameters.NUMERICAL_FEATURE, 
        parameters.CATEGORICAL_FEATURE
    ]])

# tokenize the text
text_tokenizer = text_tokenizer.TextTokenizer()
train_df[parameters.TEXT_FEATURE] = train_df[parameters.TEXT_FEATURE].fillna("")
tokenized_df = text_tokenizer.tokenize_pytorch_tensor(train_df[[parameters.TEXT_FEATURE]])
hidden_states_df = text_tokenizer.tokenized_dataframe(tokenized_df)

# combine features and targets
preprocessed_data = pd.concat(
    [
        preprocessed_num_cat_features_df, 
        hidden_states_df, 
        train_df[parameters.TARGET]
    ], 
    axis=1
)

# train model
le = LabelEncoder().fit(names)
preprocessed_data["encoded_target"] = le.transform(preprocessed_data[parameters.TARGET])

model = light_gbm_model.LightGbmClassifier().create_model()
model.fit_model(
    preprocessed_data[parameters.FEATURES], 
    preprocessed_data[parameters.TARGET]
)

# evaluate model
processed_test_df = feature_engineer.transform(test_df[
    parameters.NUMERICAL_FEATURE, 
    parameters.CATEGORICAL_FEATURE]
)
test_df[parameters.TEXT_FEATURE] = test_df[parameters.TEXT_FEATURE].fillna("")
tokenized_test_df = text_tokenizer.tokenize_pytorch_tensor(test_df[[parameters.TEXT_FEATURE]])
hidden_states_test_df = text_tokenizer.tokenized_dataframe(tokenized_test_df)
preprocessed_test_df = pd.concat(
    [
        processed_test_df, 
        hidden_states_test_df, 
        test_df[parameters.TARGET]
    ], 
    axis=1
)

# generate predictions
preprocessed_test_df = preprocessed_test_df.rename(
    columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x)
)
actual = preprocessed_test_df[parameters.TARGET].values
predictions = model.predict(preprocessed_test_df[parameters.FEATURES])
decoded_predictions = le.inverse_transform(predictions)

accuracy = accuracy_score(actual, decoded_predictions)
print(f"Accuracy: {accuracy}")