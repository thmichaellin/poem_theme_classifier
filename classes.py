import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.mixture import GaussianMixture


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
        return data.reset_index()

    def load_translate_dict(self, translate_dict: dict) -> None:
        self.translate_dict = translate_dict

    def parse_text(self) -> None:
        self.data['Poem'] = (self.data['Poem']
                             .replace(r'[\r\n]', ' ', regex=True)
                             .replace(r'\s+', ' ', regex=True)
                             .str.strip())

    def parse_tags(self) -> None:
        self.data['Tags'] = self.data['Tags'].str.replace(
            r'([^,]+),\s*([^,]+),\s*&\s*([^,]+)',
            r'\1 \2 & \3',
            regex=True).str.split(',').apply(set)

    def count_tags(self) -> None:
        self.tags = self.exploded_tags.value_counts()
        self.tags = self.tags.reset_index()
        self.tags.columns = ['Tag', 'Count']

    def translate_tags(self) -> None:
        if not self.translate_dict:
            raise ValueError("Translation dictionary is empty!")

        self.exploded_tags = self.exploded_tags.replace(self.translate_dict)
        self.data['Tags'] = self.exploded_tags.groupby(
            self.exploded_tags.index).agg(set)
        self.exploded_tags = self.data['Tags'].explode()
        self.count_tags()

    def plot_tags(self, num_tags: int = 30) -> None:
        sns.barplot(x='Tag', y='Count', data=self.tags.head(num_tags))
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Tags')
        plt.ylabel('Frequency')
        plt.title(f'Top {num_tags} Most Common Tags')
        plt.tight_layout()
        plt.show()

    def one_hot_encode_tags(self):
        mlb = MultiLabelBinarizer()
        one_hot_tags = mlb.fit_transform(self.data['Tags'])
        one_hot_encoded_df = pd.DataFrame(one_hot_tags, columns=mlb.classes_)
        return one_hot_encoded_df

    def latent_class_analysis(self, n_classes: int):
        gmm = GaussianMixture(n_components=n_classes, random_state=42)
        one_hot_data = self.one_hot_encode_tags()
        gmm.fit(one_hot_data)
        self.data['latent_class'] = gmm.predict(one_hot_data)

        means = gmm.means_
        tag_names = one_hot_data.columns
        means_df = pd.DataFrame(means, columns=tag_names)

         # For each latent class, sort tags by their mean value (higher mean = more associated with that class)
        sorted_tags_by_class = means_df.T.sort_values(by=0, ascending=False)  # Sort by means

        # Print out the top tags associated with each latent class
        for class_num in range(n_classes):
            print(f"Top tags for class {class_num}:")
            print(sorted_tags_by_class[class_num].head(100))  # Top 10 tags for the class
            print("\n")
        return gmm