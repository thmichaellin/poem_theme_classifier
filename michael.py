from classes import *


if __name__ == "__main__":
    poetry = PoetryData('./data/PoetryFoundationData.csv', ['Poem', 'Tags'])

    # print(poetry.search_poems_by_tag('Sciences'))

    poetry.load_translate_dict('cluster_tags.json')

    # poetry.plot_tags(30)
    # print(poetry.data)
    # poetry.data.to_csv('./data/data_parsed.csv', index=False)
    # poetry.gaussian_mixture(n_classes=8)
