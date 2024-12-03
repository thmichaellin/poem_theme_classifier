import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


class PoetryData:
    def __init__(self, path: str = './data/PoetryFoundationData.csv',
                 cols: list = ['Poem', 'Tags']):
        self.data = self.load_poems(path, cols)
        self.tags = Counter()
        self.count_tags()

    def load_poems(self, path: str, cols: list) -> pd.DataFrame:
        data = pd.read_csv(path, usecols=cols)
        data = data.dropna()
        data['Tags'] = data['Tags'].apply(lambda tags:
                                          tags.replace(" & ", ",").split(','))
        return data

    def count_tags(self):
        for tags in self.data['Tags']:
            self.tags.update(tags)

    def plot_tags(self):
        tag_df = pd.DataFrame(self.tags.items(), columns=['Tag', 'Count'])
        tag_df = tag_df.sort_values(by='Count', ascending=False)
        sns.barplot(x='Tag', y='Count', data=tag_df.head(30))
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Tags')
        plt.ylabel('Frequency')
        plt.show()
