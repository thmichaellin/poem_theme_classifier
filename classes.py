import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


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

    def load_translate_dict(self, json_path: str) -> None:
        with open(json_path, 'r') as f:
            cluster_tags_dict = json.load(f)
        self.translate_dict = cluster_tags_dict
        self.translate_tags()

    def parse_text(self) -> None:
        self.data['Poem'] = (self.data['Poem']
                             .replace(r'[\r\n]', ' ', regex=True)
                             .replace(r'\s+', ' ', regex=True)
                             .str.strip())

    def parse_tags(self) -> None:
        self.data['Tags'] = self.data['Tags'].str.replace(
            r'([^,]+),\s*([^,]+),\s*&\s*([^,]+)',
            r'\1_\2_and_\3',
            regex=True)
        self.data['Tags'] = self.data['Tags'].str.replace(
            r'([^,]+),\s([^,]+),\s([^,]+)',
            r'\1_\2_and_\3',
            regex=True)
        self.data['Tags'] = self.data['Tags'].str.replace(
            r'(\w+)\s*&\s*(\w+)',
            r'\1_and_\2',
            regex=True)
        self.data['Tags'] = self.data['Tags'].str.replace(
            r'\s+',
            '_',
            regex=True
        ).str.split(',').map(set)

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

    def search_poems_by_tag(self, tag: str) -> pd.DataFrame:
        return self.data[self.data['Tags'].apply(lambda tags: tag in tags)]


class PoetrySubset(PoetryData):
    def __init__(self, parent: PoetryData,
                 validation_size: float = 0.2,
                 random_state: int = 10):
        self.data = parent.data
        self.process_labels()
        self.train_data, self.val_data = train_test_split(
            self.data,
            test_size=validation_size,
            random_state=random_state
        )

    def process_labels(self):
        mlb = MultiLabelBinarizer()
        one_hot_tags = mlb.fit_transform(self.data['Tags'])

        one_hot_encoded_df = pd.DataFrame({
            'Poem': self.data['Poem'],
            'Tags': [list(row) for row in one_hot_tags]
        })
        self.data = one_hot_encoded_df


class MultiLabelDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
