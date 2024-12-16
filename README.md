# Multinomial Naive Bayes Text Classifier

## Project Overview
The **Multinomial Naive Bayes Text Classifier** project focuses on implementing a text classification model using the Naive Bayes algorithm, with both Scikit-learn and a custom-built implementation. The project demonstrates how to preprocess textual data, train the model, and evaluate its performance.

## Dataset
The project uses the `20 Newsgroups` dataset from Scikit-learn, which contains text documents categorized into 20 different topics. This dataset is widely used for text classification and clustering tasks.

## Libraries Used
- `pandas` for data manipulation
- `numpy` for numerical computations
- `matplotlib` for visualization
- `scikit-learn` for dataset loading, text vectorization, and model training

## Steps in the Project

### 1. Data Loading and Preparation
- Loaded the `20 Newsgroups` dataset using `fetch_20newsgroups`.
- Extracted text data and corresponding target labels.
- Displayed sample text data for an initial understanding.
- Created a small data frame for visualization of text-label pairs.

### 2. Text Vectorization
- Transformed the text data into a numerical format using the `TfidfVectorizer` from Scikit-learn.
- The TF-IDF representation ensures that the importance of words is weighted based on their frequency in a document and across the corpus.

### 3. Data Splitting
- Split the vectorized data into training and validation sets using an 80-20 split.

### 4. Model Training and Evaluation
- **Scikit-learn Multinomial Naive Bayes**:
  - Trained using the `MultinomialNB` class from Scikit-learn.
  - Achieved an accuracy of **81.66%** on the validation set.

- **Custom Implementation of Multinomial Naive Bayes**:
  - Built a Naive Bayes classifier from scratch using `numpy`.
  - Implemented smoothing, prior probability calculation, and likelihood estimation.
  - Achieved the same accuracy of **81.66%**, validating the custom implementation.

### 5. Result Comparison
- Compared the accuracies of the Scikit-learn model and the custom-built model using a bar chart.
- Both models achieved identical accuracies, showcasing the validity of the custom implementation.

### 6. Code Structure
- **Classes**:
  - `CustomMultinomialNB`: Implements Naive Bayes logic with smoothing, prior, and likelihood calculation.
- **Functions**:
  - `fit`: Trains the model on the provided data.
  - `predict_log_proba`: Calculates log probabilities for predictions.
  - `predict`: Predicts the class labels.
  - `score`: Computes the model’s accuracy.

### 7. Visualization
- Used a bar chart to compare model accuracies visually.
- Displayed sample text and label data for intuitive understanding.

## File Structure
- `multinomial_nb_classifier.ipynb`: Jupyter Notebook containing the entire workflow.
