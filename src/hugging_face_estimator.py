import sagemaker
from sagemaker.huggingface import HuggingFace


class HuggingFaceEstimator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.create_model()
        self.role=sagemaker.get_execution_role()
        self.hyperparameters = {
            "epochs": 1,
            "train_batch_size": 128,
            "model_name": "distilbert-base-uncased",
        }
    
    def create_model(self):
        return HugggingFace(
            entry_point="train.py",
            source_dir="s3://path_to_training.rar.gz", 
            output_path="s3://path_to_outputs",
            instance_type="ml.p3.2xlarge",
            instance_count=1,
            transformers_version="4.6",
            pytorch_version="1.7",
            py_version="py36",
            role=self.role,
            hyperparameters=self.hyperparameters
        )   

    def fit_model(self, features, target):
        return self.model.fit(features, target)