# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 20:22:57 2024

@author: Dhrumit Patel
"""

"""
Milestone Project 2: SkimLit

The purpose is to build an NLP model to make reading medical abstracts easier.
"""

# Check for GPU?
# !nvidia-smi
# !nvidia-smi -L

"""
Get the data

Since we will be replicating the paper (PubMed 200K RCT), let's download the dataset they used.

We can do so from author's github

git clone https://github.com/Franck-Dernoncourt/pubmed-rct
dir pubmed-rct 

# Check what files are in the PubMed_20K dataset
cd pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign
dir

Contains 3 files dev.txt, test.txt, train.txt
"""

# Start our experiments using the 20k dataset with numbers replaced by "@" sign
data_dir = "pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign/"

# Check all the filenames in the target directory
import os
filenames = [data_dir + filename for filename in os.listdir(data_dir)]
filenames

"""
Preprocess the data
"""

# Create a function to read the lines of a document
def get_lines(filename):
    """
    Reads filename (a text filename) and returns the lines of text as a list.
    
    Args:
        filename (str): a string containing the target filepath.
        
    Returns:
        A list of strings with one string per line from the target filename.
    """
    with open(filename, "r") as f:
        return f.readlines()

# Let's read in the training lines
train_lines = get_lines(filename=data_dir + "train.txt") # read the lines within the training file
train_lines[:20]

len(train_lines)


# Let's write a function to preprocess our data as above (List of dictionaries)
def preprocess_text_with_line_numbers(filename):
    """
    Returns a list of dictionaries of abstract line data.
    
    Takes in filename, reads its contents, and sorts through each line,
    extracting things like the target label, the text of the sentence,
    how many senetences are in the current abstract and what sentence
    number the target line is.
    """
    input_lines = get_lines(filename) # get all lines from filename
    abstract_lines = "" # Create an empty abstract
    abstract_samples = [] # Create an empty list of abstract to store dictionaries
    
    # Loop through each line in the target file
    for line in input_lines:
        if line.startswith("###"): # Check to see if the line is an ID line
            abstract_id = line
            abstract_lines = "" # Reset the abstract string if the line is an ID line
            
        elif line.isspace(): # Check to see if line is a new line
            abstract_line_split = abstract_lines.splitlines() # Split abstract into seperate lines
            
            # Iterate through each line in a single abstract and count them at the same time
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {} # Create an empty dictionary for each line
                target_text_split = abstract_line.split("\t") # Split target label from text
                line_data["target"] = target_text_split[0] # Get the target label
                line_data["text"] = target_text_split[1].lower() # Get target text and lower it
                line_data["line_number"] = abstract_line_number # What number line foes the line appear in the abstract?
                line_data["total_lines"] = len(abstract_line_split) - 1 # How many total line are there in the target abstract? (start from 0)
                abstract_samples.append(line_data) # Add line data dictionary to abstract samples list
        
        else: # If the above conditions aren't fulfilled, then the line contains a labelled sentence
            abstract_lines += line
            
    return abstract_samples
            
# Get data from file and preprocess it
train_samples = preprocess_text_with_line_numbers(filename = data_dir + "train.txt")
val_samples = preprocess_text_with_line_numbers(filename = data_dir + "dev.txt") # dev is another name for validation dataset
test_samples = preprocess_text_with_line_numbers(filename = data_dir + "test.txt")

len(train_samples), len(val_samples), len(test_samples)

# Check the first abstract of our training data
train_samples[:14]

"""
Now that our data is in the format of a list of dictionaries, How about
we turn it into a DataFrame to further visualize it?
"""
import pandas as pd
train_df = pd.DataFrame(train_samples)
val_df = pd.DataFrame(val_samples)
test_df = pd.DataFrame(test_samples)

train_df[:14]

# Distribution of labels in training data
train_df["target"].value_counts()

# Let's check length of different lines (Number of sentences per abstract (X-axis) vs Number of occurrences (Y-axis))
train_df["total_lines"].plot.hist()

"""
Get list of sentences
"""
# Convert abstract text lines into lists
train_sentences = train_df["text"].tolist()
val_sentences = val_df["text"].tolist()
test_sentences = test_df["text"].tolist()

len(train_sentences), len(val_sentences), len(test_sentences)

# View the first 10 lines of training sentences
train_sentences[:10]

"""
Making numeric labels (ML models require numeric labels)
"""
# One hot encode labels
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False) # We want non-sparse matrix
train_labels_one_hot = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
val_labels_one_hot = one_hot_encoder.transform(val_df["target"].to_numpy().reshape(-1, 1))
test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))

# Check what one hot encoded labels look like
train_labels_one_hot, val_labels_one_hot, test_labels_one_hot

"""
Label encode labels
"""
# Extract labels ("target" columns) and encode them into integers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())
val_labels_encoded = label_encoder.transform(val_df["target"].to_numpy())
test_labels_encoded = label_encoder.transform(test_df["target"].to_numpy())

# Check what label encoded labels look like
train_labels_encoded, val_labels_encoded, test_labels_encoded

# Get class names and number of classes from LabelEncoder instance
num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_
num_classes, class_names

"""
Starting a series of Modelling experiments
"""

"""
Model 0: Getting a baseline model (TF-IDF Multinomial Naive Bayes Classifier)
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Create a pipeline
model_0 = Pipeline([
    ("tf-idf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

# Fit the pipeline on the training data
model_0.fit(train_sentences, train_labels_encoded)

# Evaluate baseline model on validation dataset
model_0.score(val_sentences, val_labels_encoded)

# Make predictions using our baseline model
baseline_preds = model_0.predict(val_sentences)
baseline_preds

"""
For classification evaluation metrics (accuracy, precision, recall, f1-score)
"""
from helper_functions import calculate_results

# Calculate baselien results
baseline_results = calculate_results(y_true=val_labels_encoded, y_pred=baseline_preds)
baseline_results


train_sentences[:10]

"""
Preparing our data (the text) for deep sequence model

Before we start builidng deeper models, we had got to create vectorization and embedding layers
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# How long is each sentence on average
sent_lens = [len(sentence.split()) for sentence in train_sentences]
avg_sent_len = np.mean(sent_lens)
avg_sent_len

# What's the distribution look like?
import matplotlib.pyplot as plt
plt.hist(sent_lens, bins=20)

# How long of a sentence length covers 95% of examples?
output_seq_length = int(np.percentile(sent_lens, 95))
output_seq_length

# Maximum sequence length in the training set
max(sent_lens)

"""
Create a TextVectorizer layer

We want to make a layer which maps our texts from words to numbers
"""

# How many words are in our vocab? This is taken from Table2 from paper
max_tokens = 68000 # Came from paper by authors

# Create text vectorizer
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
text_vectorizer = TextVectorization(max_tokens=max_tokens, # Numebr of words in vocabulary
                                    output_sequence_length=output_seq_length) # Desired output length of vectorized sequences

# Adapt text vectorizer to training sentences
text_vectorizer.adapt(train_sentences)

# How many words in our training vocabulary?
rct_20k_text_vocab = text_vectorizer.get_vocabulary()
print(f"Number of words in vocab: {len(rct_20k_text_vocab)}")
print(f"Most common words in the vocab: {rct_20k_text_vocab[:5]}")
print(f"Least common words in the vocab: {rct_20k_text_vocab[-5:]}")

# Get the config of our text vectorizer
text_vectorizer.get_config()

from keras import layers
"""
Create a custom text embedding layer
"""
token_embed = layers.Embedding(input_dim=len(rct_20k_text_vocab),
                               output_dim=128, # Note: Different embedding sizes result in drastically different numbers of parameters to train
                               mask_zero=True, # Use masking to handle variable sequences lengths(save space)
                               name = "token_embedding")


"""
Creating datasets (making sure our data loads as fast as possible)

We are going to setup our data to run as fast as poccible with TensorFlow tf.data API.
"""
# Turn our data into TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_sentences, train_labels_one_hot))
valid_dataset = tf.data.Dataset.from_tensor_slices((val_sentences, val_labels_one_hot))
test_dataset = tf.data.Dataset.from_tensor_slices((test_sentences, test_labels_one_hot))

train_dataset

# Take the TensorSliceDataset's and turn them into prefetched datasets
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

train_dataset, len(train_dataset)
"""
Model 1: Conv1D with token embeddings
"""
# Create 1D Conv model to process sequences
inputs = layers.Input(shape=(1,), dtype=tf.string)
text_vectors = text_vectorizer(inputs) # Vectorize text inputs
token_embeddings = token_embed(text_vectors) # Create embedding
x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(token_embeddings)
x = layers.GlobalAveragePooling1D()(x) # Condense the ouput of our feature vector from Conv layer
outputs = layers.Dense(num_classes, activation="softmax")(x)

model_1 = tf.keras.Model(inputs, outputs)

# Compile the model
model_1.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_1.summary()

# Fit the model
history_model_1 = model_1.fit(train_dataset,
                              epochs=3,
                              steps_per_epoch=int(0.1 * len(train_dataset)), # It will only look on 10% of batches for training (to speed up training)
                              validation_data=valid_dataset,
                              validation_steps=int(0.1 * len(valid_dataset)))

# Evaluate on whole validation dataset
model_1.evaluate(valid_dataset)

# Make predictions on the validation dataset (our model predicts probabilities for each class)
model_1_pred_probs = model_1.predict(valid_dataset)
model_1_pred_probs, model_1_pred_probs.shape

# Convert pred probs to classes
model_1_preds = tf.argmax(model_1_pred_probs, axis=1)
model_1_preds
class_names
class_names[model_1_preds]

# Calculate model_1 results
model_1_results = calculate_results(y_true=val_labels_encoded, y_pred=model_1_preds)
model_1_results

"""
Model 2: Feature extraction with pretrained token embeddings

Now let's use pretrained word embeddings from TensorFlow Hub,
more sepcifically the universal sentence encoder

The paper used originally used GloVe embeddings, however we are going to stick with the later
created USE pretrained embeddings.
"""
# Download pretrained TensorFlow Hub USE
import tensorflow_hub as hub
tf_hub_embedding_layer = hub.KerasLayer("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2",
                                        trainable=False,
                                        name="universal_sentence_encoder")


"""
Building and fitting an NLP feature extraction model using pretrained embeddings TensorFlow Hub
"""
# Define feature extraction model using TF Hub layer
inputs = layers.Input(shape=[], dtype=tf.string)
pretrained_embedding = tf_hub_embedding_layer(inputs) # Tokenize text and create embedding of each sequence (512 long vector)
x = layers.Dense(128, activation="relu")(pretrained_embedding)
# Note: you could add more layers if you wanted to
outputs = layers.Dense(num_classes, activation="softmax")(x) # Create the output layer

model_2 = tf.keras.Model(inputs, outputs, name="model_2_USE_feature_extractor")

# Compile the model
model_2.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_2.summary()

# Fit model_2 to the data
with tf.device('/CPU:0'):
    history_model_2 = model_2.fit(train_dataset,
                                  epochs=3,
                                  steps_per_epoch=int(0.1 * len(train_dataset)),
                                  validation_data=valid_dataset,
                                  validation_steps=int(0.1 * len(valid_dataset)))

# Evaluate on the whole validation dataset
with tf.device('/CPU:0'):
    model_2.evaluate(valid_dataset)

# Make predictions with feature extraction model
with tf.device('/CPU:0'):
    model_2_pred_probs = model_2.predict(valid_dataset)
    model_2_pred_probs, model_2_pred_probs.shape

# Convert the prediction probabilites found with feature extraction model to labels
model_2_preds = tf.argmax(model_2_pred_probs, axis=1)
model_2_preds
class_names[model_2_preds]

# Calculate results from TF Hub pretrained embeddings results on val set
model_2_results = calculate_results(y_true=val_labels_encoded, y_pred=model_2_preds)
model_2_results

"""
Model 3: Conv1D with character embeddings

The paper which we are replicating states they used a combination of token and charcter level embeddings.
Previously, we have token level embeddings but we will need to do similar steps for characters if we want to use char-level embeddings.
"""

"""
Creating a charceter-level tokenizer
"""
train_sentences[:5]

# Make function to split sentences into characters
def split_chars(text):
    return " ".join(list(text))


# Split sequence-level data splits into character-level data splits
train_chars = [split_chars(sentence) for sentence in train_sentences]
val_chars = [split_chars(sentence) for sentence in val_sentences]
test_chars = [split_chars(sentence) for sentence in test_sentences]

train_chars, val_chars, test_chars

# What's the average character length?
char_lens = [len(sentence) for sentence in train_sentences]
mean_char_len = np.mean(char_lens)
mean_char_len

# Check the distribution of our sequences at a character-level
import matplotlib.pyplot as plt
plt.hist(char_lens, bins=7)

# Find what length of characters covers 95% of sequences
output_seq_char_len = int(np.percentile(char_lens, 95))
output_seq_char_len

# Get all keyboard characters
import string
alphabet = string.ascii_lowercase + string.digits + string.punctuation
alphabet
len(alphabet)

# Create char-level token vectorizer instances
NUM_CHAR_TOKENS = len(alphabet) + 2 # add 2 for space and OOV token (OOV = out of vocab, ['UNK])
char_vectorizer = TextVectorization(max_tokens=NUM_CHAR_TOKENS,
                                    output_sequence_length=output_seq_char_len,
                                    standardize="lower_and_strip_punctuation", # Default
                                    name="char_vectorizer")

# Adapt character vectorizer to training character
char_vectorizer.adapt(train_chars)

# Chek character vocab stats
char_vocab = char_vectorizer.get_vocabulary()
print(f"Number of different characters in character vocab: {len(char_vocab)}")
print(f"5 most common character: {char_vocab[:5]}")
print(f"5 least common characters: {char_vocab[-5:]}")

"""
Creating a character-level embedding
"""
# Create char embedding layer
char_embed = layers.Embedding(input_dim=len(char_vocab), # Number of different characters
                              output_dim=25, # This is the size of char embedding in the paper
                              mask_zero=True,
                              name="char_embed")


"""
Model 3: Building a Conv1D model to fit on character embeddings
"""
# Make Conv1D on chars only
inputs = layers.Input(shape=(1,), dtype="string")
char_vectors = char_vectorizer(inputs)
char_embeddings = char_embed(char_vectors)
x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(char_embeddings)
x = layers.GlobalMaxPool1D()(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model_3 = tf.keras.Model(inputs, outputs, name="model_3_conv1d_char_embeddings")

# Compile the model
model_3.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_3.summary()

# Create char level dataset
train_char_dataset = tf.data.Dataset.from_tensor_slices((train_chars, train_labels_one_hot)).batch(32).prefetch(tf.data.AUTOTUNE)
val_char_dataset = tf.data.Dataset.from_tensor_slices((val_chars, val_labels_one_hot)).batch(32).prefetch(tf.data.AUTOTUNE)
test_char_dataset = tf.data.Dataset.from_tensor_slices((test_chars, test_labels_one_hot)).batch(32).prefetch(tf.data.AUTOTUNE)

train_char_dataset, val_char_dataset, test_char_dataset

# Fit the model on chars only
model_3_history = model_3.fit(train_char_dataset,
                              epochs=3,
                              steps_per_epoch=int(0.1 * len (train_char_dataset)),
                              validation_data=val_char_dataset,
                              validation_steps=int(0.1 * len(val_char_dataset)))

# Evaluate the model_3
model_3.evaluate(val_char_dataset)

# Make predictions with character model only
model_3_pred_probs = model_3.predict(val_char_dataset)
model_3_pred_probs, model_3_pred_probs.shape

# Convert prediction to class labels
model_3_preds = tf.argmax(model_3_pred_probs, axis=1)
model_3_preds
class_names[model_3_preds]

# Calculate results for Conv1D model chars only
model_3_results = calculate_results(y_true=val_labels_encoded, y_pred=model_3_preds)
model_3_results

baseline_results

"""
Model 4: Combining pretrained token embeddings + characters embeddings (hybrid embedding layer)

1. Create a token level embedding model (similar to model_1)
2. Create a character level model (similar to model_3 with a slight modification)
3. Combine 1 & 2 with a concatenate (layers.Concatenate)
4. Build a series of output layer on top point 3.
5. Construct a model which takes token and character level sequences as input and produces sequence label probabilities as output.
"""

# 1. Setup token inputs/model
token_inputs = layers.Input(shape=[], dtype=tf.string, name="token_inputs")
token_embeddings = tf_hub_embedding_layer(token_inputs)
token_outputs = layers.Dense(128, activation="relu")(token_embeddings)
token_model = tf.keras.Model(inputs=token_inputs, outputs=token_outputs)

# 2. Setup char inputs/model
char_inputs = layers.Input(shape=(1,), dtype=tf.string, name="char_input")
char_vectors = char_vectorizer(char_inputs)
char_embeddings = char_embed(char_vectors)
char_bi_lstm = layers.Bidirectional(layers.LSTM(24))(char_embeddings) # bi-LSTM as given in paper
char_model = tf.keras.Model(inputs=char_inputs, outputs=char_bi_lstm)

# 3. Concatenate token and char inputs (create hybrid tokem embedding)
token_char_concat = layers.Concatenate(name="token_char_hybrid")([token_model.output, char_model.output])

# 4. Create output layers - adding in dropout (according to the paper)
combined_dropout = layers.Dropout(0.5)(token_char_concat)
combined_dense = layers.Dense(128, activation="relu")(combined_dropout)
final_dropout = layers.Dropout(0.5)(combined_dense)
output_layer = layers.Dense(num_classes, activation="softmax")(final_dropout)

# 5. Construct model with char and token inputs
model_4 = tf.keras.Model(inputs=[token_model.input, char_model.input],
                         outputs=output_layer,
                         name="model_4_token_and_char_embeddings")

# Get a summary of our model
model_4.summary()

# Plot hybrid token and character model
from keras.utils import plot_model
plot_model(model_4, show_shapes=True)

# Compile token char model
model_4.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(), # Paper says SGD optimizer
                metrics=["accuracy"])

"""
Combining token and character data into tf.data.Dataset
"""
# Combine chars and tokens into a dataset

train_char_token_data = tf.data.Dataset.from_tensor_slices((train_sentences, train_chars)) # make data
train_char_token_labels = tf.data.Dataset.from_tensor_slices(train_labels_one_hot) # make labels
train_char_token_dataset = tf.data.Dataset.zip((train_char_token_data, train_char_token_labels)) # Combine data and labels

# Prefetch and batch train data
train_char_token_dataset = train_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# For validation dataset
val_char_token_data = tf.data.Dataset.from_tensor_slices((val_sentences, val_chars))
val_char_token_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
val_char_token_dataset = tf.data.Dataset.zip((val_char_token_data, val_char_token_labels))

# Prefetch and batch val data
val_char_token_dataset = val_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Check out training char and token embedding dataset
train_char_token_dataset, val_char_token_dataset

# Fitting a model on token and character-level sequences
with tf.device('/CPU:0'):
    history_model_4 = model_4.fit(train_char_token_dataset,
                                  epochs=3,
                                  steps_per_epoch=int(0.1 * len(train_char_token_dataset)),
                                  validation_data=val_char_token_dataset,
                                  validation_steps=int(0.1 * len(val_char_token_dataset)))

# Evaluate on the whole validation dataset
with tf.device('/CPU:0'):
    model_4.evaluate(val_char_token_dataset)
    
    # Make predictions using the token-character model hybrid
    model_4_pred_probs = model_4.predict(val_char_token_dataset)
    model_4_pred_probs, model_4_pred_probs.shape
    
    # Converting to prediction probabilities to labels
    model_4_preds = tf.argmax(model_4_pred_probs, axis=1)
    model_4_preds

model_4_preds
class_names[model_4_preds]

# Get results of token char hybrid model
model_4_results = calculate_results(y_true=val_labels_encoded, y_pred=model_4_preds)
model_4_results

"""
Model 5: Transfer learning with pretrained token embeddings + character embeddings +
positional embeddings
"""
train_df.head()

"""
Create positional embeddings
"""
# How many different line numbers are there?
train_df["line_number"].value_counts()

# Check the distribution of "line_number" column
train_df["line_number"].plot.hist()

# Use TensorFlow to create one-hot encoded tensors of our "line_number" column
train_line_numbers_one_hot = tf.one_hot(train_df["line_number"].to_numpy(), depth=15)
val_line_numbers_one_hot = tf.one_hot(val_df["line_number"].to_numpy(), depth=15)
test_line_numbers_one_hot = tf.one_hot(test_df["line_number"].to_numpy(), depth=15)
train_line_numbers_one_hot[:10], train_line_numbers_one_hot.shape
train_line_numbers_one_hot[0].shape
train_line_numbers_one_hot[0].dtype

# How many different numbers of lines are there?
train_df["total_lines"].value_counts()

# Check the distribution of "total_lines" column
train_df["total_lines"].plot.hist()

# Check the coverage of a "total_lines" / What length of 95% covers our abstract string?
np.percentile(train_df["total_lines"], 98)

# Use TensorFlow One-hot encoded tensors for our "total_lines" column
train_total_lines_one_hot = tf.one_hot(train_df["total_lines"].to_numpy(), depth=20)
val_total_lines_one_hot = tf.one_hot(val_df["total_lines"].to_numpy(), depth=20)
test_total_lines_one_hot = tf.one_hot(test_df["total_lines"].to_numpy(), depth=20)
train_total_lines_one_hot[:10], train_total_lines_one_hot.shape
train_total_lines_one_hot[0].shape
train_total_lines_one_hot[0].dtype

"""
Building a tribrid embedding model

1. Create a token-level model
2. Create a character-level model
3. Create a model for the "line_number" feature
4. Create a model for the "total_lines" feature
5. Combine the outputs of 1 & 2 using tf.keras.layers.Concatenate
6. Combine the outputs of 3,4,5 using tf.keras.layers.Concatenate
7. Create an output layer to accept the tribrid embedding and output label probabilities.
8. Combine the inputs of 1,2,3,4 and outputs of 7 into tf.keras.Model
"""
# 1. Token inputs
token_inputs = layers.Input(shape=[], dtype="string", name="token_inputs")
token_embeddings = tf_hub_embedding_layer(token_inputs)
token_outputs = layers.Dense(128, activation="relu")(token_embeddings)
token_model = tf.keras.Model(inputs=token_inputs, outputs=token_outputs)

# 2. Char inputs
char_inputs = layers.Input(shape=(1,), dtype="string", name="char_inputs")
char_vectors = char_vectorizer(char_inputs)
char_embeddings = char_embed(char_vectors)
char_bi_lstm = layers.Bidirectional(layers.LSTM(24))(char_embeddings)
char_model = tf.keras.Model(inputs=char_inputs, outputs=char_bi_lstm)

# 3. Create a model for "line_number" feature
line_number_inputs = layers.Input(shape=(15,), dtype=tf.float32, name="line_number_input")
x = layers.Dense(32, activation="relu")(line_number_inputs)
line_number_model = tf.keras.Model(inputs=line_number_inputs, outputs=x)

# 4. Create a model for "total_lines" feature
total_lines_inputs = layers.Input(shape=(20,), dtype=tf.float32, name="total_lines_input")
y = layers.Dense(32, activation="relu")(total_lines_inputs)
total_lines_model = tf.keras.Model(inputs=total_lines_inputs, outputs=y)

# 5. Combine the outputs of token and char embeddings into a hybrid embedding
combined_embeddings = layers.Concatenate(name="char_token_hybrid_embedding")([token_model.output, char_model.output])
z = layers.Dense(256, activation="relu")(combined_embeddings)
z = layers.Dropout(0.5)(z)

# 6. Combine positional embedding with combined token and char embeddings
tribrid_embeddings = layers.Concatenate(name="char_token_positional_embedding")([line_number_model.output, total_lines_model.output, z])

# 7. Create output layer
output_layer = layers.Dense(num_classes, activation="softmax", name="output_layer")(tribrid_embeddings)

# 8. Put together model withall kinds of inputs
model_5 = tf.keras.Model(inputs=[line_number_model.input, 
                                 total_lines_model.input,
                                 token_model.input,
                                 char_model.input], outputs=output_layer, name="model_5_tribrid_embedding_model")

# Get a summary of our tribrid model
model_5.summary()

from tensorflow.keras.utils import plot_model
plot_model(model_5, show_shapes=True)


# Compile token char and postional embedding model
model_5.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2), # Helps to prevent overfitting
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

"""
Create tribrid embeddings datasets using tf.data
"""

# Create training and validation datasets (with all 4 kinds of input data)
train_char_token_pos_data = tf.data.Dataset.from_tensor_slices((train_line_numbers_one_hot,
                                                                train_total_lines_one_hot,
                                                                train_sentences,
                                                                train_chars))
train_char_token_pos_labels = tf.data.Dataset.from_tensor_slices(train_labels_one_hot)
train_char_token_pos_dataset = tf.data.Dataset.zip((train_char_token_pos_data, train_char_token_pos_labels))

train_char_token_pos_dataset = train_char_token_pos_dataset.batch(32).prefetch(tf.data.AUTOTUNE)


# Do the same as above for the validation dataset
val_char_token_pos_data = tf.data.Dataset.from_tensor_slices((val_line_numbers_one_hot,
                                                              val_total_lines_one_hot,
                                                              val_sentences,
                                                              val_chars))
val_char_token_pos_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
val_char_token_pos_dataset = tf.data.Dataset.zip((val_char_token_pos_data, val_char_token_pos_labels))

val_char_token_pos_dataset = val_char_token_pos_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Check the input shapes
train_char_token_pos_dataset, val_char_token_pos_dataset

# Fit the model
with tf.device('/CPU:0'):
    history_model_5 = model_5.fit(train_char_token_pos_dataset,
                                  epochs=3,
                                  steps_per_epoch=int(0.1 * len(train_char_token_pos_dataset)),
                                  validation_data=val_char_token_pos_dataset,
                                  validation_steps=int(0.1 * len(val_char_token_pos_dataset)))

with tf.device('/CPU:0'):
    # Evaluate our model on whole validation dataset
    model_5.evaluate(val_char_token_pos_dataset)
    
    # Make predictions with the char token pos model
    model_5_pred_probs = model_5.predict(val_char_token_pos_dataset)
    model_5_pred_probs, model_5_pred_probs.shape
    
    # Convert prediction probabilities to the labels
    model_5_preds = tf.argmax(model_5_pred_probs, axis=1)
    model_5_preds

model_5_preds
class_names[model_5_preds]

# Calculate results of char token pos model
model_5_results = calculate_results(y_true=val_labels_encoded, y_pred=model_5_preds)
model_5_results

"""
Compare model results
"""

# Combine model results into a dataframe
all_model_results = pd.DataFrame({"model_0_baseline": baseline_results,
                                  "model_1_custom_token_embedding": model_1_results,
                                  "model_2_pretrained_token_embedding": model_2_results,
                                  "model_3_custom_char_embedding": model_3_results,
                                  "model_4_hybrid_char_token_embedding": model_4_results,
                                  "model_5_pos_char_token_embedding": model_5_results})

all_model_results = all_model_results.transpose()
all_model_results

# Reduce the accuracy to same scale as other metrics
all_model_results["accuracy"] = all_model_results["accuracy"]/100

all_model_results

# Plot and comapre all model results
all_model_results.plot(kind="bar", figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0))

# Sort models results using f1-score
all_model_results.sort_values("f1", ascending=True)["f1"].plot(kind="bar", figsize=(10, 7))

"""
Save and load model
"""
# Save the best performing model to SavedModel format (default)
model_5.save("skimlit_tribrid_model_me")

# Load in best performing model
from keras.models import load_model
with tf.device('/CPU:0'):
    loaded_model = load_model("skimlit_tribrid_model_me")

# Make predictions with our loaded model on the validation set
with tf.device('/CPU:0'):
    loaded_pred_probs = loaded_model.predict(val_char_token_pos_dataset)
    loaded_pred_probs, loaded_pred_probs.shape

    # Convert prediction probabilities to labels
    loaded_preds = tf.argmax(loaded_pred_probs, axis=1)
    loaded_preds
    
loaded_preds[:10]
class_names[loaded_preds]

# Calculate the results of our loaded model
loaded_model_results = calculate_results(y_true=val_labels_encoded, y_pred=loaded_preds)
loaded_model_results

assert model_5_results == loaded_model_results # If nothing displays in console, it means True

# Check the loaded model summary
loaded_model.summary()


"""
Optional - for the loaded model you can use your own trained model
"""
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import os
url = "https://drive.google.com/file/d/1DYr3Ew9tU6dph_fI0JeTZ6GbdzZpWr8K/view?usp=sharing"

# Load in downloaded online model
loaded_gs_model = load_model("skimlit_tribrid_model")

# Evaluate the online loaded model
loaded_gs_model.evaluate(val_char_token_pos_dataset)
loaded_preds = tf.argmax(loaded_pred_probs, axis=1) 
loaded_preds[:10]

# Evaluate loaded model's predictions 
loaded_model_results = calculate_results(val_labels_encoded, loaded_preds) 
loaded_model_results

# Check loaded model summary
loaded_model.summary()

# Create test dataset batch and prefetched 
test_pos_char_token_data = tf.data.Dataset.from_tensor_slices((test_line_numbers_one_hot, test_total_lines_one_hot, test_sentences, test_chars)) 
test_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(test_labels_one_hot) 

test_pos_char_token_dataset = tf.data.Dataset.zip((test_pos_char_token_data, test_pos_char_token_labels)) 

test_pos_char_token_dataset = test_pos_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Make predictions on the test dataset 
with tf.device('/CPU:0'):
    test_pred_probs = loaded_model.predict(test_pos_char_token_dataset, verbose=1) 
    test_preds = tf.argmax(test_pred_probs, axis=1) 
    test_preds[:10]

# Evaluate loaded model test predictions 
loaded_model_test_results = calculate_results(y_true=test_labels_encoded, y_pred=test_preds) 
loaded_model_test_results

# Get list of class names of test predictions 
test_pred_classes = [label_encoder.classes_[pred] for pred in test_preds] 
test_pred_classes

# Create prediction-enriched test dataframe 

# Add a new column "prediction" to the test dataframe, containing predicted classes
test_df["prediction"] = test_pred_classes  

# Add a new column "pred_prob" to the test dataframe, containing the maximum prediction probability
test_df["pred_prob"] = tf.reduce_max(test_pred_probs, axis=1).numpy() 

# Add a new column "correct" to the test dataframe, which is True if the prediction matches the target, False otherwise
# This creates a binary column indicating whether the prediction is correct or not
test_df["correct"] = test_df["prediction"] == test_df["target"]  

# Display the first 20 rows of the enriched test dataframe
test_df.head(20)

# Find top 100 most wrong samples (note: 100 is an abitrary number, you could go through all of them if you wanted)
top_100_wrong = test_df[test_df["correct"] == False].sort_values("pred_prob", ascending=False)[:100] 
top_100_wrong

# Investigate top wrong predictions for rows in the top 100 wrong predictions dataframe
for row in top_100_wrong[0:10].itertuples():
    # Unpack row values
    _, target, text, line_number, total_lines, prediction, pred_prob, _ = row

    # Display information about the prediction
    print(f"Target: {target}, Pred: {prediction}, Prob: {pred_prob}, Line number: {line_number}, Total lines: {total_lines}\n")

    # Display the text associated with the prediction
    print(f"Text:\n{text}\n")

    # Separator for better readability
    print("-----------------------------------------------------------------------\n")


import json
import requests

# Download and open example abstracts (copy and pasted from PubMed)
url = "https://github.com/Dhrumit1314/Skimlit_NLP/blob/main/abstract_data.json"
response = requests.get(url)

# Check if the download was successful (status code 200)
if response.status_code == 200:
    # Load the JSON data from the response
    example_abstracts = json.loads(response.text)
    print("Example abstracts loaded successfully.")
else:
    print(f"Failed to download example abstracts. Status code: {response.status_code}")

# See what our example abstracts look like 
abstracts = pd.DataFrame(example_abstracts)
abstracts

# Import necessary library
from spacy.lang.en import English

# Setup English sentence parser with spaCy
nlp = English()

# Add the sentencizer to the spaCy pipeline
sentencizer = nlp.add_pipe("sentencizer")

# Example abstract from the loaded dataset
example_abstract = example_abstracts[0]["abstract"]
example_abstract

# Create a spaCy "doc" object by parsing the example abstract
doc = nlp(example_abstract)
doc

# Extract sentences from the spaCy doc and convert to string type
abstract_lines = [str(sent) for sent in list(doc.sents)]
# Display the detected sentences from the abstract
abstract_lines

# Get the total number of lines in the sample
total_lines_in_sample = len(abstract_lines)

# Initialize an empty list to store dictionaries containing features for each line
sample_lines = []

# Iterate through each line in the abstract and create a list of dictionaries containing features for each line
for i, line in enumerate(abstract_lines):
    # Create a dictionary to store features for the current line
    sample_dict = {}

    # Store the text of the line in the dictionary
    sample_dict["text"] = str(line)

    # Store the line number in the dictionary
    sample_dict["line_number"] = i

    # Store the total number of lines in the sample (subtracting 1 to make it 0-based index)
    sample_dict["total_lines"] = total_lines_in_sample - 1

    # Append the dictionary to the list
    sample_lines.append(sample_dict)

# Display the list of dictionaries containing features for each line
sample_lines

# Get all line_number values from the sample abstract
test_abstract_line_numbers = [line["line_number"] for line in sample_lines]

# One-hot encode to the same depth as training data, so the model accepts the right input shape
test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15)

# Display the one-hot encoded line numbers
test_abstract_line_numbers_one_hot

# Get all total_lines values from sample abstract 
test_abstract_total_lines = [line["total_lines"] for line in sample_lines] 

# One-hot encode to same depth as training data, so model accepts right input shape 
test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20) 
test_abstract_total_lines_one_hot

# Split abstract lines into characters 
abstract_chars = [split_chars(sentence) for sentence in abstract_lines] 
abstract_chars

import tensorflow as tf
import time

# Define the depths for one-hot encoding
line_numbers_depth = 15
total_lines_depth = 20 

# Prepare the input features
test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=line_numbers_depth)
test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=total_lines_depth)
test_abstract_abstract_lines = tf.constant(abstract_lines)
test_abstract_abstract_chars = tf.constant(abstract_chars)

# Make predictions on the sample abstract features
start_time = time.time()

with tf.device('/CPU:0'):
    # Note - Here you can use loaded_model if you want
    test_abstract_pred_probs = model_5.predict(x=(test_abstract_line_numbers_one_hot, test_abstract_total_lines_one_hot, tf.constant(abstract_lines), tf.constant(abstract_chars))) 

end_time = time.time()


# Display the prediction probabilities
print("Prediction Probabilities:", test_abstract_pred_probs)

# Display the time taken for predictions
print("Time taken for predictions: {:.2f} seconds".format(end_time - start_time))

# Turn prediction probabilities into prediction classes 
test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1) 
test_abstract_preds

# Turn prediction class integers into string class names
test_abstract_pred_classes = [label_encoder.classes_[i] for i in test_abstract_preds] 
test_abstract_pred_classes

# Visualize abstract lines and predicted sequence labels 
for i, line in enumerate(abstract_lines):
    print(f"{test_abstract_pred_classes[i]}: {line}")