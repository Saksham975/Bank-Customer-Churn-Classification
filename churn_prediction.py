# Dataset link - https://www.kaggle.com/santoshd3/bank-customers

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

# reading data
bank_churn_data = pd.read_csv('../dataset/Churn Modeling.csv')

# dropping unwanted columns
bank_churn_data.drop('RowNumber', axis='columns', inplace=True)  # dropping RowNumber
bank_churn_data.drop('CustomerId', axis='columns', inplace=True)  # dropping CustomerId
bank_churn_data.drop('Surname', axis='columns', inplace=True)  # dropping Surname

# data exploration
shape = bank_churn_data.shape
data_types = bank_churn_data.dtypes
null_values = bank_churn_data.isnull().sum()

# checking class distribution
churn_no = bank_churn_data[bank_churn_data.Exited == 0].Exited
churn_yes = bank_churn_data[bank_churn_data.Exited == 1].Exited
plt.xlabel('Exited')
plt.ylabel('Number of customers')
plt.title('Bank customer churn prediction')
plt.hist([churn_no, churn_yes], color=['green', 'red'], label=['Exited = No', 'Exited = Yes'])
plt.legend()
plt.show()

# visualising data
# 1. with respect to tenure
tenure_churn_no = bank_churn_data[bank_churn_data.Exited == 0].Tenure
tenure_churn_yes = bank_churn_data[bank_churn_data.Exited == 1].Tenure

plt.xlabel('Tenure')
plt.ylabel('Number of customers')
plt.title('Bank customer churn prediction')
plt.hist([tenure_churn_no, tenure_churn_yes], color=['green', 'red'], label=['Exited = No', 'Exited = Yes'])
plt.legend()
plt.show()

# 2. with respect to age
age_churn_no = bank_churn_data[bank_churn_data.Exited == 0].Age
age_churn_yes = bank_churn_data[bank_churn_data.Exited == 1].Age

plt.xlabel('Age')
plt.ylabel('Number of customers')
plt.title('Bank customer churn prediction')
plt.hist([age_churn_no, age_churn_yes], color=['green', 'red'], label=['Exited = No', 'Exited = Yes'])
plt.legend()
plt.show()

# 3. with respect to balance
balance_churn_no = bank_churn_data[bank_churn_data.Exited == 0].Balance
balance_churn_yes = bank_churn_data[bank_churn_data.Exited == 1].Balance

plt.xlabel('Balance')
plt.ylabel('Number of customers')
plt.title('Bank customer churn prediction')
plt.hist([balance_churn_no, balance_churn_yes], color=['green', 'red'], label=['Exited = No', 'Exited = Yes'])
plt.legend()
plt.show()

# 4. with respect to credit card
HasCrCard_churn_no = bank_churn_data[bank_churn_data.Exited == 0].HasCrCard
HasCrCard_churn_yes = bank_churn_data[bank_churn_data.Exited == 1].HasCrCard

plt.xlabel('Has Credit Card')
plt.ylabel('Number of customers')
plt.title('Bank customer churn prediction')
plt.hist([HasCrCard_churn_no, HasCrCard_churn_yes], color=['green', 'red'], label=['Exited = No', 'Exited = Yes'])
plt.legend()
plt.show()

# 5. with respect to salary
salary_churn_no = bank_churn_data[bank_churn_data.Exited == 0].EstimatedSalary
salary_churn_yes = bank_churn_data[bank_churn_data.Exited == 1].EstimatedSalary

plt.xlabel('Salary')
plt.ylabel('Number of customers')
plt.title('Bank customer churn prediction')
plt.hist([salary_churn_no, salary_churn_yes], color=['green', 'red'], label=['Exited = No', 'Exited = Yes'])
plt.legend()
plt.show()

# pre-processing columns
bank_churn_data['Gender'].replace({'Female': 1, 'Male': 0}, inplace=True)
bank_churn_data['Tenure'] = bank_churn_data['Tenure'] / 10
bank_churn_data['NumOfProducts'] = bank_churn_data['NumOfProducts'] / 4
bank_churn_data = pd.get_dummies(data=bank_churn_data, columns=['Geography'])

cols_to_scale = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
scaler = MinMaxScaler()
bank_churn_data[cols_to_scale] = scaler.fit_transform(bank_churn_data[cols_to_scale])

for col in bank_churn_data:
    print(f"{col} - {bank_churn_data[col].unique()}")
print()
print(bank_churn_data.sample(5))

# preparing training and testing data
X = bank_churn_data.drop('Exited', axis='columns')
y = bank_churn_data['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(bank_churn_data.shape)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# preparing ANN for classification
model = keras.Sequential([
    keras.layers.Dense(24, input_shape=(12,), activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)

y_pred = model.predict(X_test)
for i in range(0, len(y_pred)):
    if y_pred[i] < 0.5:
        y_pred[i] = 0
    else:
        y_pred[i] = 1

print(model.evaluate(X_test, y_test))

# plotting performance metrics - confusion matrix, precision, recall, f1-score
print()
print(classification_report(y_test, y_pred))

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
