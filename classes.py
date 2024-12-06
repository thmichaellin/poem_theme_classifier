import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np


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

    def train_word2vec(self, vector_size: int = 100, window: int = 5, min_count: int = 1):
        # Prepare tag sentences
        tag_sentences = self.data['Tags'].apply(list).tolist()

        # Train Word2Vec
        self.word2vec_model = Word2Vec(sentences=tag_sentences,
                                       vector_size=vector_size,
                                       window=window,
                                       min_count=min_count,
                                       workers=4)

    def cluster_tags(self, n_clusters: int = 5):
        if not self.word2vec_model:
            raise ValueError("Word2Vec model not trained yet!")

        # Get tag embeddings
        tag_vectors = {tag: self.word2vec_model.wv[tag]
                       for tag in self.word2vec_model.wv.index_to_key}
        tag_names = list(tag_vectors.keys())
        tag_embeddings = list(tag_vectors.values())

        # Cluster using KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(tag_embeddings)

        # Store clusters
        self.tag_clusters = {tag: cluster for tag,
                             cluster in zip(tag_names, cluster_labels)}

    def visualize_clusters(self):
        if not self.tag_clusters:
            raise ValueError("Tags have not been clustered yet!")

        # Get embeddings and cluster labels
        tag_vectors = {tag: self.word2vec_model.wv[tag]
                       for tag in self.word2vec_model.wv.index_to_key}
        # Convert to NumPy array
        tag_embeddings = np.array(list(tag_vectors.values()))
        cluster_labels = list(self.tag_clusters.values())

        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42,
                    perplexity=30, n_iter=1000)
        reduced_embeddings = tsne.fit_transform(
            tag_embeddings)  # This will now work

        # Plot the clusters
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                              c=cluster_labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label="Cluster")
        plt.title("Tag Clusters Visualized with t-SNE")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.show()
