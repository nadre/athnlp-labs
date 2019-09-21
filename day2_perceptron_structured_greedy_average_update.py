#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
from sklearn.metrics import accuracy_score


from athnlp.readers.brown_pos_corpus import BrownPosTag


def extract_word_features(word):
    word_features = []

    feature_functions = [
        lambda w: w.endswith('ed'),
        lambda w: w.isdigit(),
        lambda w: w[0].isupper(),
        # get creative!
    ]

    for feature_fn in feature_functions:
        if feature_fn(word):
            word_features.append(1)
        else:
            word_features.append(0)

    return word_features


def sentence_to_vectors(sentence):
    vocab_size = len(sentence.dictionary.x_dict)
    num_labels = len(sentence.dictionary.y_dict)
    feature_vectors = []
    label_vectors = []
    for i in range(len(sentence.x)):
        word_idx = sentence.x[i]
        features = np.zeros(vocab_size)
        features[word_idx] = 1
        word = sentence.dictionary.x_dict.get_label_name(word_idx)
        word_features = extract_word_features(word)

        previous_label_features = np.zeros(num_labels)

        features = np.hstack([features, word_features, previous_label_features])
        feature_vectors.append(features)

        label_vector = np.zeros(num_labels)
        label_idx = sentence.y[i]
        label_vector[label_idx] = 1
        label_vectors.append(label_vector)
    return feature_vectors, label_vectors


def prepare_data(dataset, sentence_to_vector_mapping_fn=sentence_to_vectors, cutoff=None):
    word_vectors = []
    label_vectors = []
    
    if type(cutoff) is int:
        dataset = dataset[:cutoff]
    
    for sentence in dataset:
        tmp_word_vectors, tmp_label_vectors = sentence_to_vector_mapping_fn(sentence)
        word_vectors.append(tmp_word_vectors)
        label_vectors.append(tmp_label_vectors)
    return word_vectors, label_vectors


def predict(word_vector, weight_vectors):
    dot_products = calculate_dotproducts(word_vector, weight_vectors)
    return np.argmax(dot_products)


def calculate_dotproducts(word_vector, weight_vectors):
    dot_products = []
    for weight_vector in weight_vectors:
        result = np.dot(weight_vector, word_vector)
        dot_products.append(result)
    return dot_products


def train(sentences, labels, weight_vectors):
    for feature_vectors, label_vectors in zip(sentences, labels):
        true_label_sequence = []
        predicted_label_sequence = []
        previous_label_idx = None
        for i in range(len(feature_vectors)):
            feature_vector = feature_vectors[i]

            if previous_label_idx:
                feature_vector[previous_label_idx] = 1

            label_vec = label_vectors[i]

            predicted_label_idx = predict(feature_vector, weight_vectors)

            # tricky
            previous_label_idx = len(feature_vector) - len(label_vec) + predicted_label_idx

            predicted_label_sequence.append(predicted_label_idx)

            true_label_idx = np.argmax(label_vec)
            true_label_sequence.append(true_label_idx)

            if predicted_label_sequence is not true_label_sequence:
                weight_vectors[predicted_label_idx] -= feature_vector
                weight_vectors[true_label_idx] += feature_vector

    return weight_vectors


def calculate_accuracy(sentences, labels, weight_vectors):
    all_predictions = []
    all_true_labels = []
    for word_vectors, label_vectors in zip(sentences, labels):
        all_predictions.extend([predict(wv, weight_vectors) for wv in word_vectors])
        all_true_labels.extend([np.argmax(lv) for lv in label_vectors])
    return accuracy_score(all_predictions, all_true_labels)


def run_evaluation(train_word_vecs, train_label_vecs, test_word_vecs, test_label_vecs):
    num_features = len(train_word_vecs[0][0])
    num_labels = len(train_label_vecs[0][0])
    
    print('train on train, test on test')
    train_weights = [np.zeros(num_features) for _ in range(num_labels)]
    for epoch in range(4):
        train_weights = train(train_word_vecs, train_label_vecs, train_weights)
        accuracy = calculate_accuracy(test_word_vecs, test_label_vecs, train_weights)
        print(epoch, accuracy)

    print('train on train, test on test, randomize order')
    train_weights = [np.zeros(num_features) for _ in range(num_labels)]

    zipped_samples = zip(train_word_vecs, train_label_vecs)
    random_zipped_samples = sorted(zipped_samples, key=lambda k: random.random())
    train_word_vecs_random, train_label_vecs_random = zip(*random_zipped_samples)

    for epoch in range(4):
        train_weights = train(train_word_vecs_random, train_label_vecs_random, train_weights)
        accuracy = calculate_accuracy(test_word_vecs, test_label_vecs, train_weights)
        print(epoch, accuracy)
        
    print('train on train, test on test, randomize order, each epoch')
    train_weights = [np.zeros(num_features) for _ in range(num_labels)]

    for epoch in range(4):

        zipped_samples = zip(train_word_vecs, train_label_vecs)
        random_zipped_samples = sorted(zipped_samples, key=lambda k: random.random())
        train_word_vecs_random, train_label_vecs_random = zip(*random_zipped_samples)

        train_weights = train(train_word_vecs_random, train_label_vecs_random, train_weights)
        accuracy = calculate_accuracy(test_word_vecs, test_label_vecs, train_weights)
        print(epoch, accuracy)


corpus = BrownPosTag()
cutoff = 500
test_sentences, test_labels = prepare_data(corpus.test, cutoff=cutoff)
train_sentences, train_labels = prepare_data(corpus.train, cutoff=cutoff)
run_evaluation(train_sentences, train_labels, test_sentences, test_labels)

