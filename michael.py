import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from classes import *

if __name__ == "__main__":
    poetry = PoetryData('./data/PoetryFoundationData.csv', ['Poem', 'Tags'])
    # poetry.plot_tags(40)

    # print(poetry.exploded_tags)
    # print(poetry.data)
    poetry.train_word2vec(vector_size=50, window=3, min_count=2)
    poetry.cluster_tags(n_clusters=10)
    poetry.visualize_clusters()
