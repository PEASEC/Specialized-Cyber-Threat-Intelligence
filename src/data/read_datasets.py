import json
import pandas as pd
from sklearn.model_selection import train_test_split

def read_special_cysec_data():
    UnsupportedError
    return dataset, labels

def read_general_cysec_data():
    with open("./msexchange-server-cti-dataset/external/GeneralCySec/dataset_1.json") as dataset_1_file:
        dataset_1 = json.load(dataset_1_file) 
    with open("./msexchange-server-cti-dataset/external/GeneralCySec/dataset_2.json") as dataset_2_file:
        dataset_2 = json.load(dataset_2_file) 
        
    dataset = [instance["text"] for instance in dataset_1] + [instance["text"] for instance in dataset_2]
    labels = [instance["label"] for instance in dataset_1] + [instance["label"] for instance in dataset_2]
    
    labels_binary = [1 if int(label) > 2 else 0 for label in labels]
    
    return dataset, labels_binary

def read_specialized_cysec_data():    
    df_train_full = pd.read_csv("./msexchange-server-cti-dataset/external/SpecializedCySec/df_train_full.csv")
    df_train = pd.read_csv("./msexchange-server-cti-dataset/external/SpecializedCySec/df_train.csv")
    df_dev_full = pd.read_csv("./msexchange-server-cti-dataset/external/SpecializedCySec/df_dev_full.csv")
    df_dev = pd.read_csv("./msexchange-server-cti-dataset/external/SpecializedCySec/df_dev.csv")
    df_test = pd.read_csv("./msexchange-server-cti-dataset/external/SpecializedCySec/df_test.csv")
    
    X_train_full, y_train_full = df_train_full["text"].tolist(), df_train_full["label"].tolist()
    X_train, y_train = df_train["text"].tolist(), df_train["label"].tolist()
    X_dev_full, y_dev_full = df_dev_full["text"].tolist(), df_dev_full["label"].tolist()
    X_dev, y_dev = df_dev["text"].tolist(), df_dev["label"].tolist()
    X_test, y_test = df_test["text"].tolist(), df_test["label"].tolist()
    
    return (X_train_full, y_train_full), (X_train, y_train), (X_dev_full, y_dev_full), (X_dev, y_dev), (X_test, y_test)


def split_for_normal_shot(dataset, labels, test_size=0.2, build_test_set=False):
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(dataset, labels, test_size=test_size, random_state=42)
    
    X_train_pos = [instance for i,instance in enumerate(X_train) if y_train[i] == 1]
    X_train_neg = [instance for i,instance in enumerate(X_train) if y_train[i] == 0]

    if build_test_set:
        X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, test_size=0.5, random_state=42)
        return (X_train, y_train), (X_dev, y_dev), (X_test, y_test) 
    
    return (X_train, y_train), (X_dev_test, y_dev_test)

def split_for_few_and_normal_shot(dataset, labels, few_shot_train_size=32, few_shot_dev_set=False):
    (X_train_full, y_train_full), (X_dev_full, y_dev_full), (X_test, y_test) = \
        split_for_normal_shot(dataset, labels, test_size=0.4, build_test_set=True)
    
    X_train_full_pos = [instance for i,instance in enumerate(X_train_full) if y_train_full[i] == 1]
    X_train_full_neg = [instance for i,instance in enumerate(X_train_full) if y_train_full[i] == 0]
    
    test_size_pos = (len(X_train_full_pos) - (few_shot_train_size/2)) / len(X_train_full_pos)
    test_size_neg = (len(X_train_full_neg) - (few_shot_train_size/2)) / len(X_train_full_neg)
    
    X_train_pos, X_rest_pos, y_train_pos, _ = train_test_split(
            X_train_full_pos, [1]*len(X_train_full_pos), test_size=test_size_pos, random_state=42)
    X_train_neg, X_rest_neg, y_train_neg, _ = train_test_split(
            X_train_full_neg, [0]*len(X_train_full_neg), test_size=test_size_neg, random_state=42)
    
    if few_shot_dev_set:
        devtest_size_pos = (len(X_rest_pos) - (few_shot_train_size/2)) / len(X_rest_pos)
        devtest_size_neg = (len(X_rest_neg) - (few_shot_train_size/2)) / len(X_rest_neg)
        X_dev_pos, _, y_dev_pos, _ = train_test_split(X_rest_pos, [1]*len(X_rest_pos), test_size=devtest_size_pos, random_state=42)
        X_dev_neg, _, y_dev_neg, _ = train_test_split(X_rest_neg, [0]*len(X_rest_neg), test_size=devtest_size_neg, random_state=42)
    
        return (X_train_full, y_train_full), (X_train_pos + X_train_neg, y_train_pos + y_train_neg), (X_dev_full, y_dev_full), \
            (X_dev_pos + X_dev_neg, y_dev_pos + y_dev_neg), (X_test, y_test)
    
    return (X_train_full, y_train_full), (X_train_pos + X_train_neg, y_train_pos + y_train_neg), (X_dev_full, y_dev_full), (X_test, y_test)
    
    