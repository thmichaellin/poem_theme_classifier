from classes import *

if __name__ == "__main__":
    poetry = PoetryData('./data/PoetryFoundationData.csv', ['Poem', 'Tags'])
    # poetry.plot_tags(40)

    print(poetry.exploded_tags)
    # print(poetry.data)
    gmm_model = poetry.latent_class_analysis(n_classes=20)