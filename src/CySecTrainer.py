from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
from datasets import load_metric
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import random
import time
import datetime
import os
import wandb


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
        
        
class CySecTrainer:
    def __init__(self, bert_model="roberta-large", model=None):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        if model == None:
            self.model = AutoModelForSequenceClassification.from_pretrained(bert_model, num_labels=2)
        else:
            self.model = model
    
    def train(self, X_train, y_train, X_dev, y_dev, X_test, y_test, parameters={ "seed":42, "epochs":10, "batch_size":5, "warmup_steps":200, "weight_decay":0.01, "lr":1e-5}):           
        tokenized_dataset_train = self.tokenizer(X_train, padding='max_length', truncation=True)
        tokenized_dataset_test = self.tokenizer(X_test, padding='max_length', truncation=True)
        
        train_dataset = Dataset(tokenized_dataset_train, y_train)
        test_dataset = Dataset(tokenized_dataset_test, y_test)
    
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            metrics = {
                "Accuracy: " : accuracy_score(labels, predictions),
                "F1: " : f1_score(labels, predictions, pos_label=1), 
                "Precision_1: " : recall_score(labels, predictions, pos_label=1),
                "Recall_1: " : precision_score(labels, predictions, pos_label=1),
                "Precision_0: " : recall_score(labels, predictions, pos_label=0),
                "Recall_0: " : precision_score(labels, predictions, pos_label=0),
            }
            return metrics

        
        training_args = TrainingArguments(
            output_dir= parameters["exp_dir"],
            learning_rate=parameters["lr"],
            num_train_epochs=parameters["epochs"],              
            per_device_train_batch_size=parameters["batch_size"],  
            per_device_eval_batch_size=32,   
            warmup_steps=parameters["warmup_steps"],                
            weight_decay=parameters["weight_decay"],               
            logging_dir="./logs",            
            logging_steps=parameters["eval_every"],
            seed=parameters["seed"],
        )
    
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        
        self.trainer.train()
        
    def evaluate(self):
        return self.trainer.evaluate()
    
    def save_model(self):
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
        os.mkdir("/work/scratch/tf33zuhu/few_shot/models/" + timestamp)
        self.model.save_pretrained("/work/scratch/tf33zuhu/few_shot/models/" + timestamp + "/")
        return "/work/scratch/tf33zuhu/few_shot/models/" + timestamp + "/"