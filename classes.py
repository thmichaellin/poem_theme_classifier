import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class PoetryData:
    def __init__(self, path: str = './data/PoetryFoundationData.csv',
                 cols: list = ['Poem', 'Tags']):
        self.data = self.load_poems(path, cols)
        self.translate_dict = {}
        self.parse_text()
        self.parse_tags()
        self.exploded_tags = self.data['Tags'].explode()
        self.count_tags()

    def load_poems(self, path: str, cols: list) -> pd.DataFrame:
        data = pd.read_csv(path, usecols=cols)
        data = data.dropna()
        return data

    def load_translate_dict(self, translate_dict: dict) -> None:
        self.translate_dict = translate_dict

    def parse_text(self) -> None:
        self.data['Poem'] = (self.data['Poem']
                             .replace(r'[\r\n]', ' ', regex=True)
                             .replace(r'\s+', ' ', regex=True)
                             .str.strip())

    def parse_tags(self) -> None:
        self.data['Tags'] = (self.data['Tags']
                             .str.replace(' & ', ',', regex=False)
                             .str.split(','))

    def count_tags(self) -> None:
        self.tags = self.exploded_tags.value_counts()
        self.tags = self.tags.reset_index()
        self.tags.columns = ['Tag', 'Count']

    def translate_tags(self) -> None:
        self.exploded_tags = self.exploded_tags.replace(self.translate_dict)
        self.data['Tags'] = self.exploded_tags.groupby(
            self.exploded_tags.index).apply(
                list)

    def plot_tags(self, num_tags: int = 30) -> None:
        sns.barplot(x='Tag', y='Count', data=self.tags.head(num_tags))
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Tags')
        plt.ylabel('Frequency')
        plt.title(f'Top {num_tags} Most Common Tags')
        plt.tight_layout()
        plt.show()
