import json, os

def save_datasets_for_adapet(path, X_train, y_train, X_train_augmented_pos, X_train_augmented_neg, X_dev, y_dev, X_test, y_test):
    print(os.getcwd())
    with open('./data/processed/adapet/' + path + '/val.jsonl', 'w') as outfile1, open('./src/ADAPET/data/CySec/' + path + '/val.jsonl', 'w') as outfile2:
        for i, (post, label) in enumerate(zip(X_dev, y_dev)):
            json.dump({"post":post, "idx": i, "label":label}, outfile1)
            json.dump({"post":post, "idx": i, "label":label}, outfile2)
            outfile1.write("\n")
            outfile2.write("\n")

    with open('./data/processed/adapet/' + path + '/test.jsonl', 'w') as outfile1, open('./src/ADAPET/data/CySec/' + path + '/test.jsonl', 'w') as outfile2:
        for j, (post, label) in enumerate(zip(X_test, y_test)):
            json.dump({"post":post, "idx": i+j+1, "label":label}, outfile1)
            json.dump({"post":post, "idx": i+j+1, "label":label}, outfile2)
            outfile1.write("\n")
            outfile2.write("\n")

    with open('./data/processed/adapet/' + path + '/train.jsonl', 'w') as outfile1, open('./src/ADAPET/data/CySec/' + path + '/train.jsonl', 'w') as outfile2:
        for k, (post, label) in enumerate(zip(
                X_train + X_train_augmented_pos + X_train_augmented_neg, 
                y_train + [1] * len(X_train_augmented_pos) + [0] * len(X_train_augmented_neg)
        )):
            json.dump({"post":post, "idx": i+j+k+2, "label":label}, outfile1)
            json.dump({"post":post, "idx": i+j+k+2, "label":label}, outfile2)
            outfile1.write("\n")
            outfile2.write("\n")