# Poetry Theme Prediction using DistilBERT

## Overview

This project implements a multi-label classification model using 
**DistilBERT** to predict themes for poems based on their content. 
The model is trained on the **Poetry Foundation Dataset**, which contains 
poems with associated tags that describe various aspects of the poem's themes.

### Features:
- Multi-label text classification using a pre-trained **DistilBERT** model.
- Preprocessing of text data and tag grouping with LCA.
- Visualization of model performance (losses, Hamming scores).
- Ability to predict themes for unseen poems.

## Requirements

To run this project, you need to install the required dependencies:

```bash
pip install -r requirements.txt
```

Notes:
The code for tuning DistilBERT was adapted from [this tutorial](https://colab.research.google.com/github/DhavalTaunk08/Transformers_scripts/blob/master/Transformers_multilabel_distilbert.ipynb).
All docstrings in the code were generated by ChatGPT.
