import pandas as pd
import torch

from datasets import Dataset
from transformers import AutoModel, AutoTokenizer

import parameters

class TextTokenizer:
    def __init__(
            self, 
            model_name="distilbert-base-uncased"
        ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def tokenize_pytorch_tensor(
            self,
            df
        ):

        transformer_dataset = Dataset.from_pandas(df)

        def tokenize(model_input_batch):
            return self.tokenizer(
                model_input_batch[parameters.TEXT_FEATURE],
                padding=True,
                max_length=120,
                truncation=True
            )
        tokenized_dataset = transformer_dataset.map(tokenize, batched=True, batch_size=128)
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        columns_to_remove = set(tokenized_dataset.column_names) - set(["input_ids", "attention_mask"])

        return tokenized_dataset.remove_columns(list(columns_to_remove))
    
    def extract_hidden_states(self, batch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # keys are input_ids and attention_mask
        # values are both tensor[batch_size, max_number_of_tokens_in_batch]
        inputs = {
            k: v.to(device)
            for k, v in batch.items()
        }

        #turn off gradient calculation as we don't need it
        with torch.no_grad():
            # final output of the model, the representation of the text tokens
            # use •• as the model takes input_ids and attention_mask arguments
            last_hidden_states = self.model(**inputs).last_hidden_state
            # get the CLS token which is the first one
            # [:, 0] gives a row for each batch with the first column for 768 for each
            return {"cls_hidden_state": last_hidden_states[:, 0].cpu().numpy()}
        
    def tokenized_dataframe(self, df): 
        cls_dataset = df.map(self.extract_hidden_states, batched=True, batch_size=128)
        cls_dataset.set_format(type="pandas")

        return pd.DataFrame(
            cls_dataset["cls_hidden_state"].to_list(), 
            columns=[f"feature_{n}" for n in range(1, 769)]
        )


        