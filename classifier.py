import sys
from classes import *
from helpers import *


device = 'cpu'

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',
                                                truncation=True,
                                                do_lower_case=True)
MAX_LEN = 384

model = DistilBERTClass()
model.to(device)
best_model = 'best_model.pt'
model.load_state_dict(torch.load(f'data/saved_models/{best_model}',
                                 map_location=torch.device('cpu')))
model.eval()

tags = ['Activities_and_Daily_Life', 'Arts_Culture_and_Beliefs',
        'History_and_Politics', 'Life_and_Spirituality',
        'Love_and_Romance', 'Nature_and_the_Outdoors',
        'Relationships_and_Family', 'Social_and_Cultural_Commentary']


def main():
    if not sys.stdin.isatty():
        poem_text = sys.stdin.read()
    elif len(sys.argv) > 1:
        poem_text = sys.argv[1]
    else:
        print("Error: Provide poem text as a command-line argument.")
        return

    predicted_tags = predict_tags(poem_text,
                                  model,
                                  tokenizer,
                                  MAX_LEN,
                                  device,
                                  tags,
                                  threshold=0.4)

    print(f"Predicted Tags: {predicted_tags}")


if __name__ == "__main__":
    main()
