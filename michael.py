from classes import *


if __name__ == "__main__":
    poetry = PoetryData('./data/PoetryFoundationData.csv', ['Poem', 'Tags'])

    # print(poetry.search_poems_by_tag('Sciences'))

    poetry.load_translate_dict('cluster_tags.json')

    subsets = PoetrySubset(poetry)

    print(subsets.train_data)
    print(subsets.val_data)
