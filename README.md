# tweet-emotion_classifier
This script trains a text classification model for tweet emotions using TensorFlow and Keras. It preprocesses the data, constructs a bidirectional LSTM-based neural network, and evaluates its performance on a test set.
Tweet Emotion Classifier
This project implements a text classification model for predicting emotions in tweets. It utilizes deep learning with TensorFlow and Keras to train a bidirectional LSTM-based neural network.

Features
Preprocesses tweet data using tokenization and padding.
Constructs a bidirectional LSTM-based neural network.
Trains the model on the tweet emotion dataset.
Evaluates the model's performance on a separate test set.
Displays accuracy metrics and a confusion matrix for assessment.
Usage
Install dependencies:

Copy code
pip install nlp tensorflow
Run the script:

Copy code
python tweet_emotion_classifier.py
Dataset
The tweet emotion dataset is loaded using the nlp library's load_dataset function.

License
This project is licensed under the MIT License. See the LICENSE file for details.

