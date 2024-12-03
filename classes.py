import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class PoetryData:
    def __init__(self, path: str = './data/PoetryFoundationData.csv',
                 cols: list = ['Poem', 'Tags']):
        self.data = self.load_poems(path, cols)

        self.parse_text()
        self.parse_tags()
        self.count_tags()

    def load_poems(self, path: str, cols: list) -> pd.DataFrame:
        data = pd.read_csv(path, usecols=cols)
        data = data.dropna()
        return data

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
        self.tags = self.data['Tags'].explode().value_counts()

    def plot_tags(self, num_tags: int) -> None:
        tags_df = self.tags.head(num_tags).reset_index()
        tags_df.columns = ['Tag', 'Count']
        sns.barplot(x='Tag', y='Count', data=tags_df)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Tags')
        plt.ylabel('Frequency')
        plt.title(f'Top {num_tags} Most Common Tags')
        plt.tight_layout()
        plt.show()
