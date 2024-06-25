# Sentiment Analysis with a Multi-Layer Perceptron

This project implements a sentiment analysis model using a Multi-Layer Perceptron (MLP) in TensorFlow. The model is trained on a dataset of movie reviews and aims to classify reviews as either positive or negative.

## Project Structure

- `Assignment_2_modified_ Dataset.csv`: The dataset containing movie reviews and their corresponding sentiments.
- `notebook.ipynb`: Jupyter Notebook containing the code for data preprocessing, model definition, training, and evaluation.

## Steps

1. **Data Preprocessing:**
   - Remove noise from reviews (e.g., HTML tags, special characters).
   - Convert reviews to lowercase.
   - Apply stemming and stop word removal.

2. **Data Splitting:**
   - Split the dataset into training (80%), validation (10%), and testing (10%) sets.

3. **Tokenization:**
   - Use the `Tokenizer` class from Keras to create a vocabulary from the training data.

4. **Vectorization:**
   - Convert text reviews into numerical representations using the `texts_to_matrix` method (e.g., TF-IDF).

5. **Model Definition:**
   - Define MLP classes (`MLP_1`, `MLP_2`) with different numbers of hidden layers and neurons.
   - Initialize weights and biases.
   - Implement forward pass, loss calculation, and backward pass (backpropagation).

6. **Training:**
   - Instantiate an optimizer (e.g., Adam).
   - Train the model using the training data and labels.

7. **Evaluation:**
   - Evaluate the trained model on the validation and testing sets.
   - Calculate metrics such as accuracy, precision, recall, and F1-score.

## Dependencies

- Python 3.x
- TensorFlow 2.x
- NumPy
- NLTK
- Matplotlib

## Usage

1. Install the required dependencies.
2. Run the Jupyter Notebook `notebook.ipynb` to execute the code.

## Future Work

- Experiment with different hyperparameters (e.g., learning rate, number of hidden layers, number of neurons).
- Try different text vectorization techniques (e.g., word embeddings).
- Implement regularization techniques to prevent overfitting.
