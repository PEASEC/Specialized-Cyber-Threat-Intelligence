from transformers import AutoTokenizer, GPTJForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, models
import numpy as np
import torch
import random
import pickle

def augment_data(X_train, y_train, class_to_be_augmented=1, iterations=50):
    """
    GPT-J based augmentation method
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16)
    model = model.to("cuda:0")
    
    X_train_class = [instance for i,instance in enumerate(X_train) if y_train[i] == class_to_be_augmented]
    
    template = "cyberthreat ->" if class_to_be_augmented == 1 else "other ->"
    
    augmented_data = []
    for _ in range(iterations):
        prefix_sample = random.sample(X_train_class, len(X_train_class))
        prefix_string = (template + " \n " + template + " ").join(prefix_sample) + template

        generated_tokens = model.generate(tokenizer(prefix_string, return_tensors="pt")
                                          .to("cuda:0").input_ids, do_sample=True, max_length=2048)
        generated_text = tokenizer.batch_decode(generated_tokens)[0]
        generated_text = generated_text.replace(prefix_string, "").replace("\n", "")
        generated_text = generated_text.split(template)
        augmented_data.extend([text.strip() for text in generated_text if len(text.replace(" ", "")) > 0])
    
    return augmented_data

def __get_centroid(embeddings):
    """
    function for getting the centroid of embeddings
    :param embeddings: embeddings of the data
    :return: centroid of the given embeddings
    """
    embedding_sum = np.sum(embeddings, axis=0)
    return np.divide(embedding_sum, len(embeddings[0]))


def __get_distances(embeddings, centroid, cosine=True):
    """
    pytorch based distance function that calculates the cosine or euclidean distance of all embeddings to the centroid
    :param embeddings: embeddings
    :param centroid: centroid of the embeddings from which the distance is calculated
    :param cosine: stating whether to use cosine or euclidean distance
    :return: distances of the embeddings to the centroid
    """
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    if cosine:
        return [cosine_similarity(embedding, centroid) for embedding in embeddings]
    else:
        return - [np.linalg.norm(centroid - embedding, 2) for embedding in embeddings]


def __get_k_nearest_indices(distances, k, inverse=False):
    """
    function for retrieving the k nearest indices of the distances
    :param distances: distances that were calculated before
    :param k: k
    :param inverse: boolean indicating if the sorted indices should be reversed
    :return: returns the k nearest indices of distances
    """
    distances = np.array(distances)
    if not inverse:
        return np.argsort(distances)[::-1][:k]
    else:
        return np.argsort(distances)[:k]


def __get_nearest_indices_threshold(distances, threshold, inverse=False):
    """
    function for retrieving the nearest indices of the distances from a given threshold
    :param distances: distances that were calculated before
    :param threshold: distance threshold from which the data should be included
    :param inverse: boolean indicating if all the distances from this threshold shoud be removed and if the sorted
        indices should be reversed
    :return: returns the nearest indices of the distances from a given threshold
    """
    if not inverse:
        distances = np.array(distances)
        threshold_count = len(list(filter(lambda x: x > threshold, distances)))
        return np.argsort(distances)[::-1][:threshold_count]
    else:
        distances = np.array(distances)
        threshold_count = len(list(filter(lambda x: x < threshold, distances)))
        return np.argsort(distances)[:threshold_count]


def apply_filtering(reference_data, reference_labels, unlabeled_data, reference_class=1, close_instances=False,
                    sentence_transformer="roberta-large-nli-stsb-mean-tokens", distance_threshold=0.65, verbose=False):
    """
    function that applies the filtering, which returns the unlabeled instances (from unlabeled_data) that are close
    (or distant @see :param distant_instances) to the centroid of the reference_data. The instances are sorted
    according to the distance
    :param reference_data: the data set from which the unlabeled data should be compared
    :param reference_labels: the class of the reference_data that should be used
    :param unlabeled_data: the data that should be labeled
    :param reference_class: the class that is the reference 
    :param close_instances: stating weather distant or close instances should be filtered
    :param sentence_transformer: transformer model @see https://github.com/UKPLab/sentence-transformers for more models
    :param distance_threshold: threshold defining the position from which the instances should be filtered
    :param verbose: having several print statements
    :return: returns the unlabeled instances that are close (or distant) to the reference_data (sorted by their
        distance)
    """
    verbose_print = print if verbose else lambda *a, **k: None

    verbose_print("# Loading the sentence transformer #")
    model = SentenceTransformer(sentence_transformer)
    
    reference_data = [instance for i,instance in enumerate(reference_data) if reference_labels[i] == reference_class]

    verbose_print("# Encoding reference data #")
    reference_data_embeddings = model.encode(np.array(reference_data))
    verbose_print("# Encoding unlabeled data #")
    unlabeled_data_embeddings = model.encode(np.array(unlabeled_data))

    verbose_print("# Calculating the exclusion boundary #")
    reference_data_centroid = __get_centroid(reference_data_embeddings)
    unlabeled_data_distances = __get_distances(unlabeled_data_embeddings, reference_data_centroid)
    reference_data_distances = __get_distances(reference_data_embeddings, reference_data_centroid)

    #distance_threshold = np.quantile(reference_data_distances, quantile_threshold)
    index_of_remaining_dataset = __get_nearest_indices_threshold(unlabeled_data_distances, distance_threshold,
                                                                 close_instances)

    remaining_data = [unlabeled_data[index] for index in index_of_remaining_dataset]

    return remaining_data

def save_augmented_data(data, augmented_class):
    suffix = "pos" if augmented_class == 1 else "neg"
    with open('../../msexchange-server-cti-dataset/augmented/train_' + suffix + '.pkl', 'wb') as outfile:
        pickle.dump(augmented_data, f)
        