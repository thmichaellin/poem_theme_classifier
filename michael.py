import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from classes import *

if __name__ == "__main__":
    poetry = PoetryData('./data/PoetryFoundationData.csv', ['Poem', 'Tags'])
    # poetry.plot_tags(40)

    print(poetry.exploded_tags)
    # print(poetry.data)
