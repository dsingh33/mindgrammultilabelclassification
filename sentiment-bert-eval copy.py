
"""
CREDITS: https://mccormickml.com/2019/07/22/BERT-fine-tuning/ 
"""

# PACKAGES
import pandas as pd
import numpy as np
import sklearn.metrics
import scipy.stats

import transformers as tfr
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import time
import datetime
import random

import os
import argparse


# ARGUMENTS FROM DUCTTAPE
parser = argparse.ArgumentParser(description="BERT evaluating on sentiment data")
parser.add_argument("--test", type=str, default="", help="Preprocessed test sentiment data path")
parser.add_argument("--input", type=str, default="", help="Model directory path")
parser.add_argument("--out", type=str, default="", help="Evaluation output path")
ARGS = parser.parse_args()


# GPU SETUP
device = torch.device("cuda:0")


# OTHER VARIABLES
batch_size = 32


# LOADING MODEL
input_dir = os.path.join(ARGS.input, "model_save")
bert_model = tfr.BertForSequenceClassification.from_pretrained(input_dir)
tokenizer = tfr.BertTokenizer.from_pretrained(input_dir)
# Copy the model to the GPU
bert_model.to(device)


# DATA PREPARATION FOR BERT
# test = pd.read_csv(os.path.join(ARGS.test, "test.csv"))
test = pd.read_csv("test_cts.csv")

# cleanup_nums = {"truth": {'__label__3': 0.5, '__label__5': 0.9, '__label__1': 0.1, '__label__4': 0.7, '__label__2': 0.3}}

# test.replace(cleanup_nums, inplace=True)

# Tokenizing phrases
test_tokenized = test['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# Padding tokenized phrases for equal length
max_len = 0
for i in test_tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

test_padded = np.array([i + [3]*(max_len-len(i)) for i in test_tokenized.values])

# Creating BERT inputs, attention masks, and labels
input_ids_test = torch.tensor(test_padded)

attention_mask_test = np.where(test_padded != 3, 1, 0)
attention_mask_test = torch.tensor(attention_mask_test)

labels_test = test['truth']
labels_test = torch.tensor(labels_test)

test_data = TensorDataset(input_ids_test, attention_mask_test, labels_test)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


# PREDICTION
# Put model in evaluation mode
bert_model.eval()

# Tracking variables 
predictions , true_labels, sentences, batched_sentences = [], [], [], []
val_total_loss = 0

# Predict 
for batch in test_dataloader:
  # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
  
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
  
    # Telling the model not to compute or store gradients, saving memory and 
    # speeding up prediction
    with torch.no_grad():
      # Forward pass, calculate logit predictions
        outputs = bert_model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

    logits = outputs[0]

  # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    input_ids = b_input_ids.to('cpu').numpy()

  # Turn ids back into strings
    for input_id in input_ids:
      sentences.append(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_id)))

    # Calculate the accuracy for this batch of test sentences.
    eval_loss = sklearn.metrics.mean_squared_error(label_ids, logits)
    val_total_loss += eval_loss

    # Report the MSE for this batch.
    print("")
    print("  MSE: {0:.2f}".format(eval_loss))
  
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)
    all_input_ids.append(sentences)

# Combine the predictions for each batch into a single list.
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = [item.tolist() for item in flat_predictions]
flat_predictions = [item for sublist in flat_predictions for item in sublist]

# Combine the correct labels for each batch into a single list.
flat_true_labels = [item for sublist in true_labels for item in sublist]

# Combine the correct sentences for each batch into a single list.
flat_true_sentences = [item for sublist in batched_sentences for item in sublist]


# SAVING DATA
d = {'true labels': flat_true_labels, 'predictions': flat_predictions}
df = pd.DataFrame(data=d)
df.to_csv(os.path.join(ARGS.out, "test_results.csv"))

d = {'sentences': flat_true_sentences}
df = pd.DataFrame(data=d)
df.to_csv(os.path.join(ARGS.out, "test_sentences.csv"))