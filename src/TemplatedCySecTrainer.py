from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
import torch
import numpy as np
from datasets import load_metric
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import random
import transformers
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.file_utils import PaddingStrategy
from transformers.models.bert import BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers import DataCollatorForLanguageModeling
from transformers import pipeline
from tqdm import tqdm
       
class LMdataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])  
    
    
InputDataClass = NewType("InputDataClass", Any)

"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of PyTorch/TensorFlow tensors or NumPy arrays.
"""
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, Any]])

def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ((ind,ind+sll-1))

@dataclass
class DataCollatorForFewShotLearningWithDemonstration(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }
        
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"]
        )
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch
        
        # labels = words that are masked in the input
        labels = inputs.clone()
        masked_indices = torch.full(labels.shape, 0)
        
        y = self.tokenizer("Is it about a cybersecurity threat?")["input_ids"][2:-1]
        
        for i, subtensor in enumerate(labels.tolist()):
            _, ind = find_sub_list(y, subtensor)
            masked_indices[i][ind + 1] = 1
            
        masked_indices = masked_indices.bool()
        
        labels[~masked_indices] = -100
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        return inputs, labels

    
    
class TemplatedCySecTrainer:
    def __init__(self, bert_model="roberta-large", model=None):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        if model is None:
            self.model = AutoModelForMaskedLM.from_pretrained("roberta-large")
        else:
            self.model = model
        
        
    def _template(self, instance, label):
        token_label = "Yes." if label == 1 else "No."
        text = instance + ". Is it about a cybersecurity threat? " + token_label
        return text

    def _template_test(self, instance):
        text = instance + ". Is it about a cybersecurity threat? <mask>."
        return text
    
    def _prepare_templated_demonstrations(self, X_train_pos, X_train_neg, X_test, X_dev):
        X_train_templated_complete = []
        X_test_templated_complete = []
        X_dev_templated_complete = []

        for _ in range(10):
            for instance in (X_train_pos + X_train_neg):
                if bool(random.getrandbits(1)):
                    X_train_templated_complete.append(instance + " " + random.choice(X_train_pos) 
                                                      + " " + random.choice(X_train_neg))
                else:
                    X_train_templated_complete.append(instance + " " + random.choice(X_train_neg) 
                                                      + " " + random.choice(X_train_pos))

        random.shuffle(X_train_templated_complete)

        for instance in X_test:
            if bool(random.getrandbits(1)):
                X_test_templated_complete.append(instance + " " + random.choice(X_train_pos) 
                                                 + " " + random.choice(X_train_neg))
            else:
                X_test_templated_complete.append(instance + " " + random.choice(X_train_neg) 
                                                 + " " + random.choice(X_train_pos))

        for instance in X_dev:
            if bool(random.getrandbits(1)):
                X_dev_templated_complete.append(instance + " " + random.choice(X_train_pos) 
                                                + " " + random.choice(X_train_neg))
            else:
                X_dev_templated_complete.append(instance + " " + random.choice(X_train_neg) 
                                                + " " + random.choice(X_train_pos))
                
        return X_train_templated_complete, X_dev_templated_complete, X_test_templated_complete
    
    
    def train(self, X_train, y_train, X_dev, y_dev, X_test, y_test, epochs=10, bs=5, warmup_steps=200, weight_decay=0.01):
        X_train = [self._template(instance, y_train[i]) for i, instance in enumerate(X_train)]
        X_dev = [self._template_test(instance) for instance in X_dev]
        X_test = [self._template_test(instance) for instance in X_test]   
            
        X_train_pos = [instance for i,instance in enumerate(X_train) if y_train[i] == 1]
        X_train_neg = [instance for i,instance in enumerate(X_train) if y_train[i] == 0]
            
        X_train, X_dev, X_test = \
            self._prepare_templated_demonstrations(X_train_pos, X_train_neg, X_test, X_dev)
        
        self.X_train = X_train
        self.X_dev = X_dev
        self.X_test = X_test
        self.y_train = y_train
        self.y_dev = y_dev
        self.y_test = y_test
        
        tokenized_X_train = self.tokenizer(X_train, truncation=True)
        tokenized_X_dev = self.tokenizer(X_dev, truncation=True)
        tokenized_X_test = self.tokenizer(X_test, truncation=True)
        
        train_dataset = LMdataset(tokenized_X_train)
        dev_dataset = LMdataset(tokenized_X_dev)
        test_dataset = LMdataset(tokenized_X_test)
       
        
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {
                "Accuracy: " : accuracy_score(labels, predictions),
                "F1: " : f1_score(labels, predictions, pos_label=1), 
                "Precision_1: " : recall_score(labels, predictions, pos_label=1),
                "Recall_1: " : precision_score(labels, predictions, pos_label=1),
                "Precision_0: " : recall_score(labels, predictions, pos_label=0),
                "Recall_0: " : precision_score(labels, predictions, pos_label=0),
            }

        
        training_args = TrainingArguments(
            output_dir= "./results",
            num_train_epochs=epochs,              
            per_device_train_batch_size=bs,  
            per_device_eval_batch_size=1,   
            warmup_steps=warmup_steps,                # number of warmup steps for learning rate scheduler
            weight_decay=weight_decay,               
            logging_dir="./logs",            
            logging_steps=10,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorForFewShotLearningWithDemonstration(
               tokenizer=self.tokenizer,
            )
        )
        
        trainer.train()
        
        
    def evaluate(self):
        unmasker = pipeline('fill-mask', model=self.model, tokenizer=self.tokenizer, targets=["Yes", "No"], device=0)
        transformers.logging.set_verbosity_error
        
        predicted_labels = []
        for X in tqdm(self.X_test):
            label = unmasker(X, verbose=False)[0]["token_str"].replace(" ", "")
            predicted_labels.append(label)
            
        predicted_labels_int = [1 if label == "Yes" else 0 for label in predicted_labels]
        
        return accuracy_score(self.y_test, predicted_labels_int), f1_score(self.y_test, predicted_labels_int, pos_label=1)
    
    def save_model(self):
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
        os.mkdir("../models/" + timestamp)
        self.model.save_pretrained("../models/" + timestamp + "/")
        return "../models/" + timestamp + "/"