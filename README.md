﻿# SkimLit: NLP Model for Medical Abstracts
 SkimLit is a natural language processing (NLP) project aimed at making the reading of medical abstracts more accessible. This project replicates the methodology outlined in the paper "PubMed 200K RCT: a Dataset for Sequenctial Sentence Classification in Medical Abstracts," using TensorFlow and various deep learning techniques.

# Project Overview

# **`Section 1`**

## Data Collection
- The PubMed 200K RCT dataset is obtained from the author's GitHub repository using the following commands:
```
git clone https://github.com/Franck-Dernoncourt/pubmed-rct
cd pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign
```

## Data Prepocessing
- Sentences are extracted from the dataset, and numeric labels are assigned for machine learning models.
- Three baseline models are established to set the foundation for more complex models.

## Baseline Model (Model 0)
- TF-IDF Multinomial Naive Bayes Classifier is implemented.
- Classification evaluation metrics such as accuracy, precision, recall, and F1-score are employed.

## Deep Sequence Models
### Model 1: Conv1D with Token Embeddings
- Custom TextVectorizer and text embedding layers are created.
- Data is optimized for efficiency using TensorFlow tf.data API.

### Model 2: Pretrained Token Embeddings
- Universal Sentence Encoder (USE) from TensorFlow Hub is used for feature extraction.

### Model 3: Conv1D with Character Embeddings
- Character-level tokenizer and embedding are implemented.
- Conv1D model is constructed using character embeddings.

### Model 4: Hybrid Embedding Layer
- Token and character-level embeddings are combined using layers.Concatenate.
- A model is developed to process both types of embeddings and output label probabilities.

### Model 5: Transfer Learning with Positional Embeddings
- Positional embeddings are introduced to enhance the model's understanding of the sequence.
- A tribrid embedding model is created, combining token, character, line_number, and total_lines features.

## Model Evaluation and Comparison
- Models are evaluated on various datasets to compare their performance.

## Save and Load Models
- Models are saved and loaded for future use.

## Model Loading and Evaluation
- Pre-trained models are loaded and evaluated on validation datasets.

## Test Dataset Processing and Prediction
- A test dataset is created, preprocessed, and used for making predictions with the loaded model.

## Enriching Test Dataframe with Predictions
- Predictions and additional columns are added to the test dataframe for analysis.

## Finding Top Wrong Predictions
- The top 100 most inaccurately predicted samples are identified.

## Investigating Top Wrong Predictions
- Detailed information on the top 10 wrong predictions is displayed.

# **`Section 2`**
## Example Abstracts
- Example abstracts are downloaded from a GitHub repository.

## Processing Example Abstracts with spaCy
- spaCy is used to parse sentences from example abstracts.

## One-Hot Encoding and Prediction on Example Abstracts
- Line numbers and total lines are one-hot encoded, and predictions are made using the loaded model.

## Visualizing Predictions on Example Abstracts
- Predicted sequence labels for each line in the abstract are displayed.

# Conclusion
- SkimLit provides a comprehensive exploration of NLP techniques for medical abstracts, from baseline models to sophisticated deep learning architectures. The models are evaluated, compared, and applied to real-world examples, offering insights into their strengths and limitations.

- Feel free to explore the code, experiment with different models, and contribute to the advancement of Skimlit NLP.

## Contributing
1. Fork the repository.
2. Create a new branch: `git checkout -b feature_branch`.
3. Commit your changes: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature_branch`.
5. Open a pull request.
