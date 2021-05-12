
"""
CREDITS: https://mccormickml.com/2019/07/22/BERT-fine-tuning/ 
"""

# PACKAGES
import pandas as pd
import numpy as np
import sklearn

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
parser = argparse.ArgumentParser(description="BERT finetuning on sentiment data")
parser.add_argument("--train", type=str, default="", help="Preprocessed training sentiment data path")
parser.add_argument("--validation", type=str, default="", help="Preprocessed validation sentiment data path")
parser.add_argument("--out", type=str, default="", help="Regression output path")
ARGS = parser.parse_args()


# OTHER VARIABLES
batch_size = 32
epochs = 4


# FUNCTIONS AND CLASSES
# Takes a time in seconds and returns a string hh:mm:ss
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


# GPU SETUP
device = torch.device("cuda:0")

# TRANSFORMERS MODEL INITIATION
# Setting num_labels = 1 makes this a regression model
bert_model = tfr.BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
tokenizer = tfr.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
bert_model.cuda()


# DATA PREPROCESSING FOR BERT
# train = pd.read_csv(os.path.join(ARGS.train, "train.csv"))
train = pd.read_csv("train_cts.csv")
train = train[0:100]
# validation = pd.read_csv(os.path.join(ARGS.validation, "val.csv"))
validation = pd.read_csv("val_cts.csv")
validation = validation[0:100]

# Turning categories into numbers, based on the stanford sentiment paper
# cleanup_nums = {"truth": {'__label__3': 0.5, '__label__5': 0.9, '__label__1': 0.1, '__label__4': 0.7, '__label__2': 0.3}}

# train.replace(cleanup_nums, inplace=True)
# validation.replace(cleanup_nums, inplace=True)

# Tokenizing phrases
train_tokenized = train['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
validation_tokenized = validation['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# Padding tokenized phrases for equal length
max_len = 0
for i in train_tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

train_padded = np.array([i + [3]*(max_len-len(i)) for i in train_tokenized.values])

max_len = 0
for i in validation_tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

validation_padded = np.array([i + [3]*(max_len-len(i)) for i in validation_tokenized.values])

# Creating BERT inputs, attention masks, and labels
input_ids_train = torch.tensor(train_padded)
input_ids_validation = torch.tensor(validation_padded)

attention_mask_train = np.where(train_padded != 3, 1, 0)
attention_mask_train = torch.tensor(attention_mask_train)

attention_mask_validation = np.where(validation_padded != 3, 1, 0)
attention_mask_validation = torch.tensor(attention_mask_validation)

labels_train = train['truth']
labels_train = torch.tensor(labels_train, device=device)

labels_validation = validation['truth']
labels_validation = torch.tensor(labels_validation, device=device)


# PREPARING DATA FOR TRAINING
# Create the DataLoader for our training set.
train_data = TensorDataset(input_ids_train, attention_mask_train, labels_train)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(input_ids_validation, attention_mask_validation, labels_validation)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


# TRAINING
# Loss function is automatically set to be MSE by transformers library
optimizer = tfr.AdamW(bert_model.parameters(), lr = 2e-5, eps = 1e-8)
total_steps = len(train_dataloader) * epochs
scheduler = tfr.get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []

# For each epoch...
for epoch_i in range(0, epochs):
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0

    # Put the model into training mode
    bert_model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        # As we unpack the batch, we'll also copy each tensor to the GPU 
        # using the `to` method.
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing 
        # a backward pass
        bert_model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        outputs = bert_model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
        
        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple.
        loss = outputs[0]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)            
    
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        

    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    bert_model.eval()

    # Reset total validation loss for this epoch.
    val_total_loss = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            outputs = bert_model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Calculate the loss for this batch of test sentences.
        eval_loss = sklearn.metrics.mean_squared_error(label_ids, logits)
        val_total_loss += eval_loss

        # Report the MSE for this batch.
        print("")
        print("  MSE: {0:.2f}".format(eval_loss))

    # Calculate the average MSE over the validation data.
    avg_val_loss = val_total_loss / len(validation_dataloader)
    print("")
    print("  Average MSE: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")


# SAVING MODEL
output_dir = os.path.join(ARGS.out, "model_save")
os.makedirs(output_dir, exist_ok=True)

# If saved using save_pretrained, can be reloaded using from_pretrained()
# Take care of distributed/parallel training
model_to_save = bert_model.module if hasattr(bert_model, 'module') else bert_model
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
#torch.save(args, os.path.join(output_dir, 'training_args.bin'))