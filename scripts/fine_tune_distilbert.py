import numpy as np
import pandas as pd


from datasets.filesystems import S3FileSystem
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from src import parameters, text_tokenizer, hugging_face_estimator

# load the data and create rating names category
wine_df = pd.read_csv("data/wine_data.csv")
bins = [0, 87, 94, np.inf]
names = ["neutral", "good", "excellent"]
wine_df["rating"] = pd.cut(wine_df["points"], bins, labels=names)
wine_df = wine_df[parameters.FEATURES + [parameters.TARGET]]

train_df, test_df = train_test_split(wine_df, test_size=0.2, random_state=42)

# encode the target
le = LabelEncoder().fit(names)
train_df["encoded_target"] = le.transform(train_df[parameters.TARGET])

# tokenize the text
text_tokenizer = text_tokenizer.TextTokenizer()
tokenized_train_df = text_tokenizer.tokenize_pytorch_tensor(train_df[[parameters.TEXT_FEATURE]])
s3 = S3FileSystem()
tokenized_train_df.save_to_disk("train_tokenized", fs=s3)

# fine tune the model
model = hugging_face_estimator.HuggingFaceEstimator(model_name="distilbert-base-uncased").create_model()
model.fit({
    "train": "s3://path_to_training_data"
})

# evaluate model
model = AutoModelForSequenceClassification.from_pretrained("s3://path_to_model")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
actual = test_df[parameters.TARGET].values
predictions = [
    pipe(text)
    for text in test_df[parameters.TEXT_FEATURE].values
]
prediction_labels = [int(prediction[0]["label"].split("_")[1]) for prediction in predictions]
decoded_predictions = le.inverse_transform(prediction_labels)
accuracy_score(actual, decoded_predictions)