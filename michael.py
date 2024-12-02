from project import *
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('./data/PoetryFoundationData.csv', usecols=['Poem', 'Tags'])
data = data.dropna()

tag_dict = Counter()

for tags in data['Tags']:
    tag_list = tags.replace(" & ", ",").split(',')
    tag_dict.update(tag_list)
    
#tags_sorted = dict(sorted(tag_dict.items(), key=lambda item: item[1], reverse=True))
#print(tags_sorted)

tag_df = pd.DataFrame(tag_dict.items(), columns=['Tag', 'Count'])
tag_df = tag_df.sort_values(by='Count', ascending=False)
sns.barplot(x='Tag', y='Count', data=tag_df.head(30))
plt.xticks(rotation=45, ha='right')
plt.xlabel('Tags')
plt.ylabel('Frequency')
plt.show()