import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Load Data
with open('mnist_test.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

# Remove first row headers
processed_data = np.array(data[1:], dtype=np.float32)

# Extract labels and images
labels = processed_data[:, 0]
images = processed_data[:, 1:]

# Normalize images to [0, 1] range
images /= 255.0

# Split data into 80% training and 20% testing data
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert images into pytorch tensors
train_images_pt = torch.tensor(train_images.reshape(-1, 1, 28, 28))
test_images_pt = torch.tensor(test_images.reshape(-1, 1, 28, 28))

# Convert labels into long tensors
train_labels_pt = torch.tensor(train_labels).long()
test_labels_pt = torch.tensor(test_labels).long()

# Create datasets
train_dataset = torch.utils.data.TensorDataset(train_images_pt, train_labels_pt)
test_dataset = torch.utils.data.TensorDataset(test_images_pt, test_labels_pt)

# Streamline batch processing and shuffling
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.convolutional_layer = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        
        # Adjust the size of the fully connected layer
        self.full_hidden_layer = nn.Linear(32 * 14 * 14, 100)  # Size after pooling

        # Output layer for 10 classes
        self.full_output_layer = nn.Linear(100, 10)

    def forward(self, input_data):
        input_data = F.relu(F.max_pool2d(self.convolutional_layer(input_data), 2))  # Pooling halves the size
        input_data = input_data.view(-1, 32 * 14 * 14)  # Adjust flattening
        input_data = torch.sigmoid(self.full_hidden_layer(input_data))
        input_data = self.full_output_layer(input_data)
        return F.log_softmax(input_data, dim=1)

net = ConvNN()
# Set hyperparameters
learning_rate = 0.01
momentum = 0.1
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

training_accuracies = []
testing_accuracies = []
num_epochs = 200  

for epoch in range(num_epochs):
    running_loss = 0.0

    # Set net to training mode
    net.train() 
    for inputs, labels in train_loader:
        inputs, labels = inputs.float(), labels.long()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = F.nll_loss(outputs, labels)
        #loss = F.mse_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Compute training accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.float(), labels.long()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    train_accuracy = 100 * correct / total
    training_accuracies.append(train_accuracy)

    # Compute testing accuracy
    correct = 0
    total = 0
    # Set the network to evaluation mode
    net.eval()  
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.float(), labels.long()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    testing_accuracies.append(test_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs} - Training Accuracy: {train_accuracy}, Testing Accuracy: {test_accuracy}")

# Compute confusion matrix after all epochs
test_preds, test_labels_list = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.float(), labels.long()
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_preds.extend(predicted.numpy())
        test_labels_list.extend(labels.numpy())

test_conf_matrix = confusion_matrix(test_labels_list, test_preds)
print("Confusion Matrix after all epochs:")
print(test_conf_matrix)

plt.plot(training_accuracies, label='Training Accuracy')
plt.plot(testing_accuracies, label='Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.show()
