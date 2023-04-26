import json
import os
import platform
import time

import pandas as pd
from pytwitter import Api
# pip install python-twitter-v2
from sklearn.model_selection import train_test_split


# Setting the paths for the datasets
DATASETS = os.path.join(os.environ["PROJECT_ROOT"], "msexchange-server-cti-dataset", "external")
GENERALCYSEC = os.path.join(DATASETS, "CySecAlert", "twitter-data")
SPECIALIZEDECYSEC = os.path.join(DATASETS, "SpecializedCySec")

TIMEOUT = 2  # at least 2

client = Api(
    consumer_key="Add your own api keys here",
    consumer_secret="Add your own api keys here",
    access_token="Add your own api keys here",
    access_secret="Add your own api keys here"
)

# Another authentication option
# client = Api('Bearer Token')

def twitter_api_call(lst):
    """
    Calling the twitter api with all ids in lst
    :param lst: List with twitter ids for the twitter api call
    :return: the tweets that could be collected from the ids
    """
    global TIMEOUT
    tweets = {}
    # One call can only contain 100 ids
    # If there are more then 100 ids in parameter lst, sepereate them into chunks of 100
    if len(lst) > 100:
        chunks = [lst[x:x + 100] for x in range(0, len(lst), 100)]
        tweets = []
        for chunk in chunks:
            try:
                response = client.get_tweets(chunk, return_json=True)
                tweets.append(response)
                # time.sleep(TIMEOUT)
            except Exception as e:
                # If there are to many requests, wait and try again 
                if isinstance(e.args[0], dict):
                    if e.args[0]["title"] == "Too Many Requests":
                        TIMEOUT = TIMEOUT + 1
                        time.sleep(10)
                        response = client.get_tweets(chunk, return_json=True)
                        tweets.append(response)
                        time.sleep(TIMEOUT)
    else:
        try:
            tweets = client.get_tweets(lst, return_json=True)
            time.sleep(TIMEOUT)
        except Exception as e:
            if isinstance(e.args[0], dict):
                if e.args[0]["title"] == "Too Many Requests":
                    TIMEOUT = TIMEOUT + 1
                    time.sleep(10)
                    tweets = client.get_tweets(lst, return_json=True)
                    time.sleep(TIMEOUT)

    return tweets


def read_general_cysec_data():
    """
    Reads the data from the CYSECALERT Dataset
    :return: returns the collected text of the tweets and the according labels
    """
    with open(os.path.join(GENERALCYSEC, "labeledtweets1.json")) as dataset_1_file:
        dataset_1 = json.load(dataset_1_file)

    with open(os.path.join(GENERALCYSEC, "labeledtweets2.json")) as dataset_2_file:
        dataset_2 = json.load(dataset_2_file)

    # Prepare datasets
    tweet_ids = []
    texts = []
    labels = []
    for tweet in dataset_1 + dataset_2:
        # If the datasets only contains ids and not text, add the ids to a list
        if "text" not in tweet:
            tweet_ids.append(str(tweet["tweetId"]))
        else:
            texts.append(tweet["text"])
            labels.append(tweet["label"])
    # if there are ids in the list call the twitter api
    if len(tweet_ids) > 0:
        tweets = twitter_api_call(tweet_ids)
        if isinstance(tweets, dict):
            tweets = tweets["data"]
        else:
            tweets2 = []
            for d in tweets:
                tweets2 = tweets2 + d["data"]
            tweets = tweets2
        # Retrieve only the important data
        texts = [item["text"] for item in tweets]
        ids = [tweet["id"] for tweet in tweets]
        not_found = [i for i in tweet_ids if i not in ids]
        print(f"GeneralCySec: {len(not_found)} of {len(dataset_1 + dataset_2)} tweets were not accessible ")
        # Get the labels from the datasets
        labels = []
        i = 0
        j = 0
        while i < len(tweets) and j < len(tweet_ids):
            if tweets[i]["id"] == tweet_ids[j]:
                if i < len(dataset_1):
                    label = 1 if int(dataset_1[i]["label"]) > 2 else 0
                else:
                    label = 1 if int(dataset_2[i - len(dataset_1)]["label"]) > 0 else 0
                labels.append(label)
                i += 1
                j = i
            else:
                j += 1

    return texts, labels


def read_specialized_cysec_data():
    """
    Read the SPECIALIZEDCYSEC dataset from the paper of this code
    :return: all texts and according labels already splitted
    """
    dataframes = {
        "df_train_full": [pd.read_csv(os.path.join(SPECIALIZEDECYSEC, "df_train_full.csv"))],
        "df_train": [pd.read_csv(os.path.join(SPECIALIZEDECYSEC, "df_train.csv"))],
        "df_dev_full": [pd.read_csv(os.path.join(SPECIALIZEDECYSEC, "df_dev_full.csv"))],
        "df_dev": [pd.read_csv(os.path.join(SPECIALIZEDECYSEC, "df_dev.csv"))],
        "df_test": [pd.read_csv(os.path.join(SPECIALIZEDECYSEC, "df_test.csv"))]
    }
    keys = list(dataframes.keys())
    # if the datasets only contains ids and not tex, collect them from th twitter api
    if ("text" not in dataframes[keys[0]][0].columns or "text" not in dataframes[
        keys[1]][0].columns or "text" not in dataframes[keys[2]][0].columns or
            "text" not in dataframes[keys[3]][0].columns or "text" not in dataframes[keys[4]][0].columns):
        # Iterating through all datasets
        for key, value in dataframes.items():
            tweet_ids = []
            rows = value[0].iterrows()
            for i, row in rows:
                tweet_ids.append(str(row["id"]))
            tweets = twitter_api_call(tweet_ids)
            if isinstance(tweets, dict):
                tweets = tweets["data"]
            else:
                tweets2 = []
                for d in tweets:
                    tweets2 = tweets2 + d["data"]
                tweets = tweets2
            # Gathering labels
            ids = [item["id"] for item in tweets]
            not_found = [i for i in tweet_ids if i not in ids]
            rows = value[0].iterrows()
            labels = []
            for i, row in rows:
                if str(row["id"]) not in not_found:
                    labels.append(row["label"])
            # Save in the existing dicionary 
            dataframes[key].append([tweet["text"] for tweet in tweets])
            dataframes[key].append(labels)
            print(f"{key}: {len(not_found)} Tweets of {len(value[0]['id'].tolist())} tweets were not accessible")
    else:
        for key, value in dataframes.items():
            dataframes[key].append(value[0]["text"].tolist())
            dataframes[key].append(value[0]["label"].tolist())

    keys = list(dataframes.keys())
    return (dataframes[keys[0]][1], dataframes[keys[0]][2]), \
    (dataframes[keys[1]][1], dataframes[keys[1]][2]), \
    (dataframes[keys[2]][1], dataframes[keys[2]][2]), \
    (dataframes[keys[3]][1], dataframes[keys[3]][2]), \
    (dataframes[keys[4]][1], dataframes[keys[4]][2])


def split_for_normal_shot(dataset, labels, test_size=0.2, build_test_set=False):
    """
    Split datasets for a typical training with much data
    :param dataset: dataset that should be splitted
    :param labels: lables according to the dataset
    :param test_size: size of the testsplit
    :param build_test_set: boolean for deciding to build test set or not
    :return: Splitted datasets
    """
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(dataset, labels, test_size=test_size, random_state=42)

    X_train_pos = [instance for i, instance in enumerate(X_train) if y_train[i] == 1]
    X_train_neg = [instance for i, instance in enumerate(X_train) if y_train[i] == 0]

    if build_test_set:
        X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, test_size=0.5, random_state=42)
        return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)

    return (X_train, y_train), (X_dev_test, y_dev_test)


def split_for_few_and_normal_shot(dataset, labels, few_shot_train_size=32, few_shot_dev_set=False):
    """
    Split datasets for few_shot and a typical training with much data
    :param dataset: dataset that should be splitted
    :param labels: lables according to the dataset
    :param few_shot_train_size: size of few shot dataset
    :param few_shot_dev_set: booleand for deciding if a dev set should be created or not
    :return: splitted datasets
    """
    (X_train_full, y_train_full), (X_dev_full, y_dev_full), (X_test, y_test) = \
        split_for_normal_shot(dataset, labels, test_size=0.4, build_test_set=True)

    X_train_full_pos = [instance for i, instance in enumerate(X_train_full) if y_train_full[i] == 1]
    X_train_full_neg = [instance for i, instance in enumerate(X_train_full) if y_train_full[i] == 0]

    test_size_pos = (len(X_train_full_pos) - (few_shot_train_size / 2)) / len(X_train_full_pos)
    test_size_neg = (len(X_train_full_neg) - (few_shot_train_size / 2)) / len(X_train_full_neg)

    X_train_pos, X_rest_pos, y_train_pos, _ = train_test_split(
        X_train_full_pos, [1] * len(X_train_full_pos), test_size=test_size_pos, random_state=42)
    X_train_neg, X_rest_neg, y_train_neg, _ = train_test_split(
        X_train_full_neg, [0] * len(X_train_full_neg), test_size=test_size_neg, random_state=42)

    if few_shot_dev_set:
        devtest_size_pos = (len(X_rest_pos) - (few_shot_train_size / 2)) / len(X_rest_pos)
        devtest_size_neg = (len(X_rest_neg) - (few_shot_train_size / 2)) / len(X_rest_neg)
        X_dev_pos, _, y_dev_pos, _ = train_test_split(X_rest_pos, [1] * len(X_rest_pos), test_size=devtest_size_pos,
                                                      random_state=42)
        X_dev_neg, _, y_dev_neg, _ = train_test_split(X_rest_neg, [0] * len(X_rest_neg), test_size=devtest_size_neg,
                                                      random_state=42)

        return (X_train_full, y_train_full), (X_train_pos + X_train_neg, y_train_pos + y_train_neg), (
            X_dev_full, y_dev_full), \
               (X_dev_pos + X_dev_neg, y_dev_pos + y_dev_neg), (X_test, y_test)

    return (X_train_full, y_train_full), (X_train_pos + X_train_neg, y_train_pos + y_train_neg), (
        X_dev_full, y_dev_full), (X_test, y_test)


def count_labels_specialized():
    datasets = {
        "df_train_full": pd.read_csv(os.path.join(SPECIALIZEDECYSEC, "df_train_full.csv")),
        "df_train": pd.read_csv(os.path.join(SPECIALIZEDECYSEC, "df_train.csv")),
        "df_dev_full": pd.read_csv(os.path.join(SPECIALIZEDECYSEC, "df_dev_full.csv")),
        "df_dev": pd.read_csv(os.path.join(SPECIALIZEDECYSEC, "df_dev.csv")),
        "df_test": pd.read_csv(os.path.join(SPECIALIZEDECYSEC, "df_test.csv")),
    }
    for key, item in datasets.items():
        print(f"Size of {key}: ", len(item.index))
        print(f"Relevant count {key}: ", len(item[item["label"] == 1].index))
        print(f"Not relevant count {key}: ", len(item[item["label"] == 0].index))


if __name__ == "__main__":

    dataset, labels_binary= read_general_cysec_data()
    if len(dataset) != len(labels_binary):
        print(len(dataset))
        print(len(labels_binary))
    if 4 in labels_binary or 3 in labels_binary or 2 in labels_binary:
        print(labels_binary)
    print(len(labels_binary))
    print(len([label for label in labels_binary if label == 1]))

    (X_train_full, y_train_full), (X_train, y_train), (X_dev_full, y_dev_full), (X_dev, y_dev), (
        X_test, y_test) = read_specialized_cysec_data()
    if len(X_train_full) != len(y_train_full):
        print(len(X_train_full))
        print(len(y_train_full))
    if len(X_train) != len(y_train):
        print(len(X_train))
        print(len(y_train))
    if len(X_dev_full) != len(y_dev_full):
        print(len(X_dev_full))
        print(len(y_dev_full))
    if len(X_dev) != len(y_dev):
        print(len(X_dev))
        print(len(y_dev))
    if len(X_test) != len(y_test):
        print(len(X_test))
        print(len(y_test))
    print(type(X_train[0]))
    print(X_train)
