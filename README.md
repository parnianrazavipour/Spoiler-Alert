# Spoiler-Alert
Final project for the NLP course in Spring 2024 at the Computer Engineering Department, Sharif University of Technology, under the supervision of Dr. Asgari.

### Disclaimer
I cloned the project from the original repository and made some changes to the code and the README file. The original repository can be found [here](https://github.com/parnianrazavipour/Introduction-to-Bioinformatics-project/tree/main) which is currently Private.
The work is done by our team and I was allowed to use the code and the README file for my own repository.

# Project Overview

This repository contains the code and models for a project that addresses two core tasks in natural language processing for movie reviews: spoiler detection and spoiler-free text generation. Using the IMDB Spoiler dataset, our goal is to accurately classify reviews as spoilers or non-spoilers and generate concise, coherent spoiler-free summaries of movie plots. The project utilizes various transformer-based models to achieve high performance in both tasks.

The repository is structured into two main folders: one for Classification (spoiler detection) and another for Generation (spoiler-free plot generation).

1. Classification Task

In this task, we aim to classify movie reviews as either spoilers or non-spoilers. We leveraged a combination of feature extraction techniques and advanced deep learning models for this task. Our method involves extracting several key features from the dataset, such as:

The presence of certain spoiler-related keywords (e.g., "die", "kill")
Sentence length and structure
Review metadata such as the number of votes or review date (used in feature engineering)
After preprocessing the data and extracting features, we trained multiple models, including LSTM, BERT, and RoBERTa. We also experimented with model ensemble techniques, such as voting between different classifiers, to further improve accuracy.

2. Generation Task

In this task, we generate spoiler-free summaries of movie plots using various transformer-based models. The dataset contains two columns: one with the full plot (containing spoilers) and another with the spoiler-free summary. The goal is to fine-tune models that can generate a coherent, spoiler-free version of the plot.

We experimented with the following transformer models:

BART, 
LED, 
BigBird, 
LongT5

These models were fine-tuned on the IMDB Spoiler dataset to generate concise, non-spoiler summaries while retaining the essence of the movie’s storyline.


# Project Results

In the generation task, our models—BART, LED, BigBird, and LongT5—successfully generated spoiler-free summaries and reviews, maintaining the plot’s essence without revealing key details. In the classification task, using LSTM, BERT, and RoBERTa, we accurately detected spoiler reviews, achieving high precision and recall. These results highlight the effectiveness of our approach in both tasks.

# Requirements

To run the scripts, ensure you have the following installed:

Python 3.8+

Pytorch

Transformers library (HuggingFace)

TensorFlow

Scikit-learn


# Assignments:

Classification: Ali Nikkhah, Sarina Zahedi, Ramtin Khoshnevis

Generation: Abolfazl Eshagh, Mobina Salimipanah, Parnian Razavi
