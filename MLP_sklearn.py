from matplotlib.pyplot import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix


# load train set data minus targets and normalize

data = np.array(
    pd.read_csv('.../medical_mnist.csv'))

# Extract labels and images
target = data[:, 0]
data= data[:, 1:]

# Standardize the features
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split the data into training and test sets
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=2)

# Define the neural network model
model = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic',
                      max_iter=50, solver='sgd', verbose=10, 
                      random_state=1, learning_rate_init=0.01, momentum=0.01)

# Initialize the weights of the neural network randomly
weights_output = np.random.rand(6, 50 + 1) * 0.1 - 0.05

# Train the neural network model
model.fit(data_train, target_train)

# Test the neural network model
target_pred = model.predict(data_test)

# Calculate the recall value
recall = recall_score(target_test, target_pred, average='weighted')
print("Recall: ", recall*100)

# Calculate the accuracy of the model
accuracy = accuracy_score(target_test, target_pred)
print("Accuracy: ", accuracy*100)


# plt.figure(figsize=(12, 6))
# plt.plot(model.n_iter_, ':', linewidth=2, label="Training Accuracy")
# plt.plot(model.n_iter_, linewidth=2, label="Testing Accuracy")
# plt.legend()
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Training and Testing Accuracy Over Epochs")
# plt.show()

# Create the confusion matrix
cm = confusion_matrix(target_test, target_pred)
plt.figure(figsize=(12, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

