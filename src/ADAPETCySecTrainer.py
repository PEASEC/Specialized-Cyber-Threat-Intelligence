from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
from datasets import load_metric
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import random
from ADAPET.src.train import train as adapet_train
from ADAPET.src.test import test as adapet_test
from ADAPET.src.utils.Config import Config
import os
from pathlib import Path
import glob
import datetime as dt
import wandb
import json
import data.read_datasets
import inspect

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
        
        
class ADAPETCySecTrainer:
    def __init__(self, config_path, kwargs=None, use_saved_model=False, model_dir=None):
        self.config = Config(config_path, kwargs, mkdir=True)
        # for i in inspect.getmembers(self.config):
        #    print(i)
        self.use_saved_model = use_saved_model
        self.model_dir = model_dir

    def train(self, return_model=False):
        if return_model:
            return adapet_train(self.config, self.use_saved_model, self.model_dir, return_model)
        else:
            adapet_train(self.config, self.use_saved_model, self.model_dir, return_model)
            return None

    def eval_model(self, path):
        # Prepare the needed variables for the calculations
        results = []
        with open(path, "r") as f:
            for line in f:
                results.append(json.loads(line))
        results_binary = [1 if item["label"] == "true" else 0 for item in results]

        if("GeneralCySec" in self.config.dataset):
            print("Reading general dataset")
            dataset, labels = data.read_datasets.read_general_cysec_data()
            (X_train_finetuning, y_train_finetuning), (X_dev_finetuning, y_dev_finetuning), (X_test_finetuning, y_test_finetuning) = \
                data.read_datasets.split_for_normal_shot(dataset, labels, build_test_set=True)
        elif("SpecializedCySec" in self.config.dataset):
            print("Reading specialized dataset")
            (X_train_full, y_train_full), (X_train_few, y_train_few), (X_dev_full, y_dev_full), (X_dev_few, y_dev_few), (X_test, y_test) = \
            data.read_datasets.read_specialized_cysec_data()
            y_test_finetuning=y_test
        else:
            print("Couldn't read dataset")
            return
        
        # Calculate scores
        accuracy = accuracy_score(y_test_finetuning, results_binary)
        f1 = f1_score(y_test_finetuning, results_binary, pos_label=1)
        recall = recall_score(y_test_finetuning, results_binary)
        precision = precision_score(y_test_finetuning, results_binary)
    
        print('Model evaluation')
        print(f'Accuracy: {accuracy}')
        print(f'F1-Score: {f1}')
        print(f'Recall: {recall}')
        print(f'Precision: {precision}')

    def evaluate(self):
        # First let adapet function predict labels, then evaluate the model 
        adapet_test(self.config.exp_dir)
        self.eval_model(self.config.exp_dir+"/test.json")
    
    def save_model(self):
        # Model is already saved by ADAPET - just returning the right folder
        return self.config.exp_dir
