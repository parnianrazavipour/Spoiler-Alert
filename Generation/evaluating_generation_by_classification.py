import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Trainer, TrainingArguments
import gc
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

movie_details = pd.read_json('IMDB_movie_details.json', lines=True)
reviews = pd.read_json('IMDB_reviews.json', lines=True)

movie_details = movie_details[movie_details['plot_synopsis'] != ""]

reviews = reviews[reviews['review_summary'] != '']


review1 = reviews.loc[1]
plot1 = movie_details.loc[1]
print(review1)
print()
print(plot1)

movie_details = movie_details.drop(columns = ['movie_id', 'duration', 'genre', 'rating', 'release_date'])
reviews = reviews.drop(columns = ['review_date', 'movie_id', 'user_id', 'is_spoiler', 'rating'])

from sklearn.model_selection import train_test_split
plot_dataset = movie_details[['plot_summary', 'plot_synopsis']]

X = plot_dataset['plot_synopsis']
y = plot_dataset['plot_summary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

train_set = pd.DataFrame({'plot_synopsis': X_train, 'plot_summary': y_train})
test_set = pd.DataFrame({'plot_synopsis': X_test, 'plot_summary': y_test})
val_set = pd.DataFrame({'plot_synopsis': X_val, 'plot_summary': y_val})


len(train_set), len(test_set), len(val_set)


train_set.loc[166]['plot_synopsis']

# Load BART tokenizer and model
model_name = "facebook/bart-large"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Tokenize data
train_encodings = tokenizer(train_set['plot_synopsis'].tolist(), truncation=True, padding="max_length", max_length=512, return_tensors="pt")
train_labels = tokenizer(train_set['plot_summary'].tolist(), truncation=True, padding="max_length", max_length=128, return_tensors="pt")
val_encodings = tokenizer(val_set['plot_synopsis'].tolist(), truncation=True, padding="max_length", max_length=512, return_tensors="pt")
val_labels = tokenizer(val_set['plot_summary'].tolist(), truncation=True, padding="max_length", max_length=128, return_tensors="pt")
test_encodings = tokenizer(test_set['plot_synopsis'].tolist(), truncation=True, padding="max_length", max_length=512, return_tensors="pt")
test_labels = tokenizer(test_set['plot_summary'].tolist(), truncation=True, padding="max_length", max_length=128, return_tensors="pt")


# Create dataset class
class PlotDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels['input_ids'][idx].clone().detach()
        return item

train_dataset_prepared = PlotDataset(train_encodings, train_labels)
val_dataset = PlotDataset(val_encodings, val_labels)
test_dataset = PlotDataset(test_encodings, test_labels)

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_prepared,
    eval_dataset=val_dataset,
)

trainer.train()


import torch, gc

gc.collect()
torch.cuda.empty_cache()

torch.save(model.state_dict, 'bart_plot_summary_generation10.pth')

model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# val_dataset = PlotDataset(test_encodings, test_labels)
test_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

all_predictions = []
all_references = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model.generate(input_ids, max_length=64, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        all_predictions.extend(outputs)
        all_references.extend(labels)

# Decode predictions and references
decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in all_predictions]
decoded_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in all_references]


# Compute metrics
from datasets import load_metric
# rouge_types=["rouge1", "rouge2", "rougeL"]
rouge = load_metric("rouge")
# rouge = rouge(rouge_types)
bleu = load_metric("bleu")
from bert_score import score


rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
bleu_result = bleu.compute(predictions=[pred.split() for pred in decoded_preds], references=[[ref.split()] for ref in decoded_labels])
P, R, F1 = score(decoded_preds, decoded_labels, lang='multilingual')
bertscore_result = {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}

results = {"bleu": bleu_result, "rouge": rouge_result, "bertscore": bertscore_result}

# Print results
from prettytable import PrettyTable

def print_results(results):
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]

    bleu_scores = results['bleu']
    table.add_row(["BLEU Score", f"{bleu_scores['bleu']:.4f}"])
    table.add_row(["Brevity Penalty", f"{bleu_scores['brevity_penalty']:.4f}"])
    table.add_row(["Length Ratio", f"{bleu_scores['length_ratio']:.4f}"])
    table.add_row(["Translation Length", bleu_scores['translation_length']])
    table.add_row(["Reference Length", bleu_scores['reference_length']])
    for i, precision in enumerate(bleu_scores['precisions'], 1):
        table.add_row([f"Precisions {i}-gram", f"{precision:.4f}"])

    rouge_scores = results['rouge']
    for key in rouge_scores:
        score = rouge_scores[key].mid
        table.add_row([f"ROUGE-{key} (Precision)", f"{score.precision:.4f}"])
        table.add_row([f"ROUGE-{key} (Recall)", f"{score.recall:.4f}"])
        table.add_row([f"ROUGE-{key} (F-Measure)", f"{score.fmeasure:.4f}"])

    bert_scores = results['bertscore']
    table.add_row(["BERTScore Precision", f"{bert_scores['precision']:.4f}"])
    table.add_row(["BERTScore Recall", f"{bert_scores['recall']:.4f}"])
    table.add_row(["BERTScore F1", f"{bert_scores['f1']:.4f}"])

    print(table)

print_results(results)


input_plot = test_set.loc[266]
input_plot['plot_summary']


# Example article for prediction
example_article = input_plot['plot_synopsis']
input_encoding = tokenizer(example_article, return_tensors="pt", max_length=256, padding="max_length", truncation=True)

print("input plot:")
print(example_article)

print()

# Generate title
model.eval()
with torch.no_grad():
    input_encoding = {k: v.to(device) for k, v in input_encoding.items()}
    outputs = model.generate(input_encoding['input_ids'], max_length=64, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    predicted_title = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Predicted plot summary:")
print(predicted_title)



########################################################################
reviews_df = pd.read_json('IMDB_reviews.json' , lines=True)
reviews_df.head()

import pandas as pd

print("Total reviews:", len(reviews_df))
print("Number of spoiler reviews:", reviews_df['is_spoiler'].sum())
print("Number of non-spoiler reviews:", len(reviews_df) - reviews_df['is_spoiler'].sum())

reviews_df['review_length'] = reviews_df['review_text'].apply(len)
print("Average review length:", reviews_df['review_length'].mean())
print("Median review length:", reviews_df['review_length'].median())

print(reviews_df['is_spoiler'].value_counts(normalize=True))

reviews_df = reviews_df[:175000]

len(set(reviews_df['movie_id']))

reviews_df['review_length'] = reviews_df['review_text'].apply(len)


import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Review length distribution
sns.histplot(reviews_df['review_length'], bins=50, kde=True)
plt.title('Distribution of Review Lengths')
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.show()

# Box plot for review lengths by spoiler
sns.boxplot(x='is_spoiler', y='review_length', data=reviews_df)
plt.title('Review Length by Spoiler Status')
plt.xlabel('Is Spoiler')
plt.ylabel('Review Length')
plt.show()

data = reviews_df[['review_text', 'is_spoiler']]
data.head()

data.shape[0]


def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.replace('\n', ' ')  # Replace newlines with spaces
    text = text.replace('\r', ' ')  # Replace carriage returns with spaces
    text = ''.join([c if c.isalnum() or c.isspace() else ' ' for c in text])  # Remove special characters
    text = ' '.join(text.split())  # Remove extra spaces
    return text

# Apply text cleaning to the dataset
data['review_text'] = data['review_text'].apply(clean_text)


from sklearn.model_selection import train_test_split
train_data, temp_data = train_test_split(data, test_size=0.25, random_state=42)

# Split the remainder into test and validation sets
test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Display the sizes of each dataset
print("Training set size:", len(train_data))
print("Validation set size:", len(val_data))
print("Test set size:", len(test_data))


from datasets import Dataset
import torch

train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset_lstm = Dataset.from_pandas(test_data)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

vocab_size = 8000
max_length = 2000  # Maximum review length
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_data['review_text'])

def tokenize_and_pad(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences


X_train = tokenize_and_pad(train_data['review_text'])
X_val = tokenize_and_pad(val_data['review_text'])
X_test = tokenize_and_pad(test_data['review_text'])

y_train = train_data['is_spoiler']
y_val = val_data['is_spoiler']
y_test = test_data['is_spoiler']

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from tensorflow.keras.layers import Bidirectional

model_lstm = Sequential([
    Embedding(vocab_size, 32, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),  # Using a bidirectional LSTM
    Dropout(0.5),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


model_lstm.build(input_shape=(None, max_length))

model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.summary()



history = model_lstm.fit(X_train, y_train, epochs=8, batch_size=32, validation_data=(X_val, y_val))


# Save the model in HDF5 forma
model_lstm.save('my_fine_tuned_model.h5')

import gc
gc.collect()
torch.cuda.empty_cache()


# Evaluate the model on the test set
loss, accuracy = model_lstm.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

from sklearn.metrics import classification_report
import numpy as np


predictions = model_lstm.predict(X_test)
predictions = (predictions > 0.5).astype(int)  # Threshold predictions for binary classification


report = classification_report(y_test, predictions, target_names=['Non-Spoiler', 'Spoiler'])
print(report)

def prepare_input(text, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded

def predict_spoiler(text, tokenizer, model, max_length=1500):
    prepared_text = prepare_input(text, tokenizer, max_length)
    prediction = model.predict(prepared_text)
    is_spoiler = (prediction > 0.5).astype(int)  # Threshold the prediction
    return bool(is_spoiler)

# Test the function
input_text = "Jack Ryan is on a working vacation in London with his family. He has retired from the CIA and is a Professor at the US Naval Academy. He is seen delivering a lecture at the Royal Naval Academy in London.Meanwhile, Ryan's wife Cathy and daughter Sally are sightseeing near Buckingham Palace. Sally and Cathy come upon a British Royal Guard, and Sally tries to get the guard to react by doing an improvised tap dance in front of him. She's impressed when the guard, trained to ignore distraction, doesn't react at all, and they leave."
is_spoiler = predict_spoiler(input_text, tokenizer, model)
print(f"The text is a {'spoiler' if is_spoiler else 'non-spoiler'}.")


# Test the function
input_text = "Three years have passed since John Brennan (Russel Crowe) and his wife Lara (Elizabeth Banks) lost their son Luke (Tyler Simpkins) in a car accident. Three years later, John is a community college teacher who is teaching English. He tries to manage his job and raising Luke."
is_spoiler = predict_spoiler(input_text, tokenizer, model)
print(f"The text is a {'spoiler' if is_spoiler else 'non-spoiler'}.")

# Test the function
input_text = "Three years have passed since John Brennan (Russel Crowe) and his wife Lara (Elizabeth Banks) lost their son Luke (Tyler Simpkins) in a car accident. Three years later, John is a community college teacher who is teaching English. He tries to manage his job and raising Luke."
is_spoiler = predict_spoiler(input_text, tokenizer, model)
print(f"The text is a {'spoiler' if is_spoiler else 'non-spoiler'}.")



indexes = list(test_dataset.index.values)


total = 0
non_spoiler = 0
for i in indexes:
    total += 1
    input_plot = test_set.loc[i]
    input_plot['plot_summary']

    # Example article for prediction
    example_article = input_plot['plot_synopsis']
    input_encoding = tokenizer(example_article, return_tensors="pt", max_length=256, padding="max_length", truncation=True)

    print("input plot:")
    print(example_article)

    print()

    # Generate title
    model.eval()
    with torch.no_grad():
        input_encoding = {k: v.to(device) for k, v in input_encoding.items()}
        outputs = model.generate(input_encoding['input_ids'], max_length=64, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        predicted_title = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Predicted plot summary:")
    print(predicted_title)
    is_spoiler = predict_spoiler(predicted_title, tokenizer, model_lstm)
    if is_spoiler != True:
        non_spoiler += 1


ratio = non_spoiler/total

print("-------------------------------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------------------------------")
print()
print("The ratio of non-spoiler plot summary generations based on the LSTM classifier is:")
print(ratio)
        
        
    

