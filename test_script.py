from classes import *


if __name__ == "__main__":
    poetry = PoetryData('./data/PoetryFoundationData.csv', ['Poem', 'Tags'])

    # print(poetry.search_poems_by_tag('Sciences'))

    print(poetry.tags)

    # poetry.load_translate_dict('./data/cluster_tags.json')

    # tags = poetry.data['Tags']

    # sum = sum(len(tag_set) for tag_set in tags)

    # print(sum / len(tags))

    # print(poetry.data.iloc[9753]['Poem'])

    # print(poetry.data.iloc[9753]['Tags'])
