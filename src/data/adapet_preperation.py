import json
import os

# Setting the paths for the datasets
MSEXCHANGE_PATH = os.path.join(os.environ["PROJECT_ROOT"], "msexchange-server-cti-dataset", "external")
#DATASETS = os.path.join("/home/tf33zuhu/Specialized-Cyber-Threat-Intelligence/msexchange-server-cti-dataset", "external")
ADAPET_PATH = os.path.join(os.environ["PROJECT_ROOT"],"src", "ADAPET")


def save_datasets_for_adapet(path, X_train, y_train, X_train_augmented_pos, X_train_augmented_neg, X_dev, y_dev, X_test,
                             y_test):
    """
    Save the datasts and the augmentations for usage from ADAPET
    """
    processed_saving_path = os.path.join(MSEXCHANGE_PATH, "processed", "adapet", path)
    adapet_saving_path = os.path.join(ADAPET_PATH, "data", "CySec", path)
    os.makedirs(processed_saving_path, exist_ok=True)
    os.makedirs(adapet_saving_path, exist_ok=True)
    with open(os.path.join(processed_saving_path, 'val.jsonl'), 'w') as outfile1, open(
            os.path.join(adapet_saving_path, 'val.jsonl'), 'w') as outfile2:
        for i, (post, label) in enumerate(zip(X_dev, y_dev)):
            json.dump({"post": post, "idx": i, "label": int(label)}, outfile1)
            json.dump({"post": post, "idx": i, "label": int(label)}, outfile2)
            outfile1.write("\n")
            outfile2.write("\n")

    with open(os.path.join(processed_saving_path, 'test.jsonl'), 'w') as outfile1, open(
            os.path.join(adapet_saving_path, 'test.jsonl'), 'w') as outfile2:
        for j, (post, label) in enumerate(zip(X_test, y_test)):
            json.dump({"post": post, "idx": i + j + 1, "label": int(label)}, outfile1)
            json.dump({"post": post, "idx": i + j + 1, "label": int(label)}, outfile2)
            outfile1.write("\n")
            outfile2.write("\n")

    with open(os.path.join(processed_saving_path, 'train.jsonl'), 'w') as outfile1, open(
            os.path.join(adapet_saving_path, 'train.jsonl'), 'w') as outfile2:
        for k, (post, label) in enumerate(zip(
                X_train + X_train_augmented_pos + X_train_augmented_neg,
                y_train + [1] * len(X_train_augmented_pos) + [0] * len(X_train_augmented_neg))):
            json.dump({"post": post, "idx": i + j + k + 2, "label": int(label)}, outfile1)
            json.dump({"post": post, "idx": i + j + k + 2, "label": int(label)}, outfile2)
            outfile1.write("\n")
            outfile2.write("\n")
