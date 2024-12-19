import logging
import json
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import DistilBertModel, DistilBertTokenizer

# Suppress warnings and set logging level
import warnings
warnings.simplefilter('ignore')
logging.basicConfig(level=logging.ERROR)


class PoetryData:
    """
    A class for loading and processing poetry data, including tag parsing,
    text cleaning, and visualizations.

    Attributes:
    ----------
    data : pd.DataFrame
        DataFrame containing the poems and their associated tags.
    translate_dict : dict
        Dictionary for translating tags.
    exploded_tags : pd.Series
        Exploded version of the tags for individual tag analysis.
    tags : pd.DataFrame
        DataFrame containing tag counts.

    Methods:
    -------
    load_poems(path: str, cols: list) -> pd.DataFrame:
        Loads poetry data from a CSV file.
    load_translate_dict(json_path: str) -> None:
        Loads a translation dictionary from a JSON file.
    parse_text() -> None:
        Cleans and preprocesses the poem text.
    parse_tags() -> None:
        Cleans and preprocesses the tags.
    count_tags() -> None:
        Counts the occurrences of each tag.
    translate_tags() -> None:
        Translates tags using the loaded translation dictionary.
    plot_tags(num_tags: int = 30) -> None:
        Plots the most common tags as a bar chart.
    search_poems_by_tag(tag: str) -> pd.DataFrame:
        Searches for poems that contain a specific tag.
    """

    def __init__(self, path: str = './data/PoetryFoundationData.csv',
                 cols: list = ['Poem', 'Tags']):
        """
        Initializes the PoetryData class by loading and processing the
        poetry data.

        Parameters:
        ----------
        path : str, optional
            Path to the CSV file containing the poetry data
            (default is './data/PoetryFoundationData.csv').
        cols : list, optional
            List of columns to load from the CSV file
            (default is ['Poem', 'Tags']).
        """

        self.data = self.load_poems(path, cols)
        self.translate_dict = {}
        self.parse_text()
        self.parse_tags()
        self.exploded_tags = self.data['Tags'].explode()
        self.count_tags()

    def load_poems(self, path: str, cols: list) -> pd.DataFrame:
        """
        Loads poetry data from a CSV file.

        Parameters:
        ----------
        path : str
            Path to the CSV file containing the poetry data.
        cols : list
            List of columns to load from the CSV file.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the poems and their tags.
        """

        data = pd.read_csv(path, usecols=cols).dropna()
        return data.reset_index()

    def load_translate_dict(self, json_path: str) -> None:
        """
        Loads a translation dictionary from a JSON file.

        Parameters:
        ----------
        json_path : str
            Path to the JSON file containing the translation dictionary.
        """

        with open(json_path, 'r') as f:
            self.translate_dict = json.load(f)
        self.translate_tags()

    def parse_text(self) -> None:
        """
        Cleans and preprocesses the poem text by removing extra whitespace and
        newline characters.
        """

        self.data['Poem'] = (self.data['Poem']
                             .replace(r'[\r\n]', ' ', regex=True)
                             .replace(r'\s+', ' ', regex=True)
                             .str.strip())

    def parse_tags(self) -> None:
        """
        Cleans and preprocesses the tags by replacing commas and spaces with
        underscores and standardizing tag formats.
        """

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
        """
        Counts the occurrences of each tag in the dataset and stores the
        result in the 'tags' attribute.
        """

        self.tags = self.exploded_tags.value_counts().reset_index()
        self.tags.columns = ['Tag', 'Count']

    def translate_tags(self) -> None:
        """
        Translates the tags using the loaded translation dictionary.
        """

        if not self.translate_dict:
            raise ValueError("Translation dictionary is empty!")
        self.exploded_tags = self.exploded_tags.replace(self.translate_dict)
        self.data['Tags'] = self.exploded_tags.groupby(
            self.exploded_tags.index).agg(set)
        self.exploded_tags = self.data['Tags'].explode()
        self.count_tags()

    def plot_tags(self, num_tags: int = 30) -> None:
        """
        Plots the most common tags as a bar chart.

        Parameters:
        ----------
        num_tags : int, optional
            The number of most common tags to display in the plot
            (default is 30).
        """

        sns.barplot(x='Tag', y='Count', data=self.tags.head(num_tags))
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Tags')
        plt.ylabel('Frequency')
        plt.title(f'Top {num_tags} Most Common Tags')
        plt.tight_layout()
        plt.show()

    def search_poems_by_tag(self, tag: str) -> pd.DataFrame:
        """
        Searches for poems that contain a specific tag.

        Parameters:
        ----------
        tag : str
            The tag to search for.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the poems that have the specified tag.
        """

        return self.data[self.data['Tags'].apply(lambda tags: tag in tags)]


class PoetrySubset(PoetryData):
    """
    A class for creating subsets of poetry data for training, validation,
    and testing.

    Inherits from the PoetryData class.

    Attributes:
    ----------
    train_data : pd.DataFrame
        DataFrame containing the training data.
    val_data : pd.DataFrame
        DataFrame containing the validation data.
    test_data : pd.DataFrame
        DataFrame containing the test data.

    Methods:
    -------
    process_labels() -> None:
        Processes the labels (tags) for multi-label classification.
    """

    def __init__(self, parent: PoetryData,
                 test_size: float = 0.3,
                 random_state: int = 10):
        """
        Initializes the PoetrySubset class by splitting the data into training,
        validation, and test subsets.

        Parameters:
        ----------
        parent : PoetryData
            The parent PoetryData object from which the data is derived.
        test_size : float, optional
            The proportion of the dataset to include in the test subset
            (default is 0.3).
        random_state : int, optional
            The random seed used for splitting the dataset
            (default is 10).
        """

        self.data = parent.data
        self.process_labels()
        self.train_data, self.val_test_data = train_test_split(
            self.data,
            test_size=test_size,
            random_state=random_state
        )
        self.val_data, self.test_data = train_test_split(
            self.val_test_data,
            test_size=.5,
            random_state=random_state
        )

    def process_labels(self) -> None:
        """
        Processes the tags for multi-label classification using
        one-hot encoding.
        """

        mlb = MultiLabelBinarizer()
        one_hot_tags = mlb.fit_transform(self.data['Tags'])

        self.data = pd.DataFrame({
            'Poem': self.data['Poem'],
            'Tags': [list(row) for row in one_hot_tags]
        })


class MultiLabelDataset(Dataset):
    """
    A custom PyTorch Dataset for handling multi-label poetry data.

    Attributes:
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer used for tokenizing the poem text.
    data : pd.DataFrame
        The poetry data (poem texts and associated tags).
    text : pd.Series
        The poem text data.
    targets : pd.Series
        The tag data.
    max_len : int
        The maximum length for tokenized input sequences.

    Methods:
    -------
    __len__() -> int:
        Returns the number of examples in the dataset.
    __getitem__(index: int) -> dict:
        Returns a single example from the dataset.
    """

    def __init__(self, dataframe: pd.DataFrame, tokenizer: DistilBertTokenizer,
                 max_len: int):
        """
        Initializes the MultiLabelDataset.

        Parameters:
        ----------
        dataframe : pd.DataFrame
            DataFrame containing the poem texts and tags.
        tokenizer : DistilBertTokenizer
            The tokenizer used for tokenizing the poem text.
        max_len : int
            The maximum length for tokenized input sequences.
        """

        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = self.data['Poem']
        self.targets = self.data['Tags']
        self.max_len = max_len

    def __len__(self) -> int:
        """
        Returns the number of examples in the dataset.

        Returns:
        -------
        int
            The number of examples in the dataset.
        """

        return len(self.text)

    def __getitem__(self, index: int) -> dict:
        """
        Returns a single example from the dataset.

        Parameters:
        ----------
        index : int
            The index of the example to retrieve.

        Returns:
        -------
        dict
            A dictionary containing the input ids, attention mask,
            token type ids, and targets for the example.
        """

        text = str(self.text[index])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
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


class DistilBERTClass(torch.nn.Module):
    """
    A custom PyTorch model class using DistilBERT for multi-label
    classification.

    Methods:
    -------
    forward(input_ids: torch.Tensor, attention_mask: torch.Tensor) ->
        torch.Tensor:
        Defines the forward pass of the model.
    """

    def __init__(self):
        """
        Initializes the DistilBERT model with additional layers for
        classification.
        """
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 8)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Parameters:
        ----------
        input_ids : torch.Tensor
            The tokenized input ids.
        attention_mask : torch.Tensor
            The attention mask to indicate padded tokens.

        Returns:
        -------
        torch.Tensor
            The output logits for multi-label classification.
        """

        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
