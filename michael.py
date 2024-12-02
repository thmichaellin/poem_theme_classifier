from project import *
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

data = pd.read_csv("./data/PoetryFoundationData.csv")

print(data.head())

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english")

inputs = tokenizer(data["Poem"][0], return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]
