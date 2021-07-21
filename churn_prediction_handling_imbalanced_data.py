# Dataset link - https://www.kaggle.com/santoshd3/bank-customers
# We will try various techniques to handle imbalanced data


import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)


# creating function for model creation
def ANN(X_train, y_train, X_test, y_test, epochs, loss):
    model = keras.Sequential([
        keras.layers.Dense(24, input_shape=(12,), activation='relu'),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs)

    print(model.evaluate(X_test, y_test))
    y_preds = model.predict(X_test)
    y_preds = np.round(y_preds)

    print("Classification Report: \n", classification_report(y_test, y_preds))
    return y_preds


# reading data
bank_churn_data = pd.read_csv('../dataset/Churn Modeling.csv')

# dropping unwanted columns
bank_churn_data.drop('RowNumber', axis='columns', inplace=True)  # dropping RowNumber
bank_churn_data.drop('CustomerId', axis='columns', inplace=True)  # dropping CustomerId
bank_churn_data.drop('Surname', axis='columns', inplace=True)  # dropping Surname

# pre-processing columns
bank_churn_data['Gender'].replace({'Female': 1, 'Male': 0}, inplace=True)
bank_churn_data['Tenure'] = bank_churn_data['Tenure'] / 10
bank_churn_data['NumOfProducts'] = bank_churn_data['NumOfProducts'] / 4
bank_churn_data = pd.get_dummies(data=bank_churn_data, columns=['Geography'])

cols_to_scale = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
scaler = MinMaxScaler()
bank_churn_data[cols_to_scale] = scaler.fit_transform(bank_churn_data[cols_to_scale])

# data exploration
shape = bank_churn_data.shape
data_types = bank_churn_data.dtypes
null_values = bank_churn_data.isnull().sum()
count_class_0, count_class_1 = bank_churn_data.Exited.value_counts()

bank_churn_data_class_0 = bank_churn_data[bank_churn_data.Exited == 0]
bank_churn_data_class_1 = bank_churn_data[bank_churn_data.Exited == 1]
print(bank_churn_data_class_0.shape)
print(bank_churn_data_class_1.shape)

# 1. Undersampling data
bank_churn_data_under_sampled_class_0 = bank_churn_data_class_0.sample(count_class_1)
bank_churn_data_under_sampled = pd.concat([bank_churn_data_under_sampled_class_0, bank_churn_data_class_1], axis=0)
print("After Under-Sampling -")
print(bank_churn_data_under_sampled.shape)
print(bank_churn_data_under_sampled.Exited.value_counts())
X = bank_churn_data_under_sampled.drop('Exited', axis='columns')
y = bank_churn_data_under_sampled.Exited
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

print(X.shape)
print(y.shape)

y_preds = ANN(X_train, y_train, X_test, y_test, 100, 'binary_crossentropy')
# f1-score for class-1 improves significantly but resuces for class-0. This indicates that we have more generalized
# classifier which classifies both classes with similar prediction score


# 2. Oversampling data
bank_churn_data_over_sampled_1 = bank_churn_data_class_1.sample(count_class_0,
                                                                replace=True)  # replace=True indicates duplicating

print(bank_churn_data_over_sampled_1.shape)
print(bank_churn_data_class_0.shape)

bank_churn_data_over_sampled = pd.concat([bank_churn_data_class_0, bank_churn_data_over_sampled_1], axis=0)

print("After over-sampling -")
print(bank_churn_data_over_sampled.Exited.value_counts())

X = bank_churn_data_over_sampled.drop('Exited', axis='columns')
y = bank_churn_data_over_sampled.Exited
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
y_preds = ANN(X_train, y_train, X_test, y_test, 100, 'binary_crossentropy')

# 3. SMOTE - Also a technique of over-sampling the data
X = bank_churn_data.drop('Exited', axis='columns')
y = bank_churn_data.Exited
smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=0, stratify=y_sm)
y_preds = ANN(X_train, y_train, X_test, y_test, 100, 'binary_crossentropy')

# 4. Ensemble with under-sampling
X = bank_churn_data.drop('Exited', axis='columns')
y = bank_churn_data.Exited
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


def get_training_batch(df_majority, df_minority, start, end):
    df_train = pd.concat([df_majority[start:end], df_minority], axis=0)
    X_train = df_train.drop('Exited', axis='columns')
    y_train = df_train.Exited
    return X_train, y_train


df = X_train.copy()
df['Exited'] = y_train
df_class0 = df[df.Exited == 0]
df_class1 = df[df.Exited == 1]

X_train, y_train = get_training_batch(df_class0, df_class1, 0, 1630)
y_pred1 = ANN(X_train, y_train, X_test, y_test, 100, 'binary_crossentropy')

X_train, y_train = get_training_batch(df_class0, df_class1, 1630, 3260)
y_pred2 = ANN(X_train, y_train, X_test, y_test, 100, 'binary_crossentropy')

X_train, y_train = get_training_batch(df_class0, df_class1, 3260, 4890)
y_pred3 = ANN(X_train, y_train, X_test, y_test, 100, 'binary_crossentropy')

X_train, y_train = get_training_batch(df_class0, df_class1, 4890, 6520)
y_pred4 = ANN(X_train, y_train, X_test, y_test, 100, 'binary_crossentropy')

y_pred_final = y_pred1.copy()
for i in range(0, len(y_pred1)):
    n_ones = y_pred1[i] + y_pred2[i] + y_pred3[i] + y_pred4[i]
    if n_ones > 2:
        y_pred_final[i] = 1
    else:
        y_pred_final[i] = 0

print("Final classification report after ensemble - ")
print(classification_report(y_test, y_pred_final))
