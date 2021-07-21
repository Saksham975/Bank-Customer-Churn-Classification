## Bank-Customer-Churn-Classification
## Creating classification model using neural networks

Dataset link - https://www.kaggle.com/santoshd3/bank-customers

Code was writtien in python on Pycharm IDE.
Libraries used - pandas, sklearn, matplotlib, tensorflow,seaborn

In this project, I have build a classification model on bank customer churn data.

Following steps were performed - 
1. Data exploration
2. Data preparation
3. Data visualisation
4. Splitting data into train-test data
5. Preparing classification model using tensorflow
6. Anlysing evaluation metrics - accuracy, preciion, recall, f1-score

Model gave an accuracy of around 85% on training data as well as test data.

Classification model features - 
1. input shape - (12,)
2. 2 hidden layers
3. 'relu' activation function was used in hidden layers
4. Dropout ratio of 0.3 and 0.2 were used in different hidden layers
5. 'adam' was used as optimizer
6. As it is binary classification, 'binary_crossentropy' was used as loss function.

Deep learning is all about trial and error.
Try to play with more number of hidden layers, different activation functions, dropout ratios, optimizer etc.
In every different implementation of code, you'll get different accuracy, preciion, recall. Try to find the best suited value as per the requirement.

# Dropout layer was used as the data was imbalanced and also to reduce the problem of overfitting.






# churn_prediction_handling_imbalanced_data.py

This file handles the imbalanced dataset and applies various techniques such as under-sampling, over-sampling, SMOTE, ensemble to overcome the problem.

Imbalanced dataset handling metrics comparison.docx - This file contains the summarization of tasks completed in handling imbalanced dataset.
