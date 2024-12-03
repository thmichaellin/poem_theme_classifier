import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from classes import *

if __name__ == "__main__":
    poetry = PoetryData('./data/PoetryFoundationData.csv', ['Poem', 'Tags'])
    # poetry.plot_tags(30)
    print(poetry.tags)
    # print(poetry.data)

# Explore colinearity between labels.
