import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from classes import *

if __name__ == "__main__":
    poetry = PoetryData()
    poetry.plot_tags()
