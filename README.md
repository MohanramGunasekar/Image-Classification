# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Classifying the MNIST Fashion dataset which has 70000 images of 10 classes of clothing using Convolutional Neural Network enabling it to identify newly prompted image.

## Neural Network Model

![Screenshot 2025-03-24 094536](https://github.com/user-attachments/assets/38f778af-bd96-4973-9470-5730eb499cd5)

## DESIGN STEPS
## STEP 1: Problem Statement
Define the objective of classifying fashion items (T-shirts, trousers, dresses, shoes, etc.) using a Convolutional Neural Network (CNN).

## STEP 2: Dataset Collection
Use the Fashion-MNIST dataset, which contains 60,000 training images and 10,000 test images of various clothing items.

## STEP 3: Data Preprocessing
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

## STEP 4: Model Architecture
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers to extract features and classify clothing items.

## STEP 5: Model Training
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

## STEP 6: Model Evaluation
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

## STEP 7: Model Deployment & Visualization
Save the trained model, visualize predictions, and integrate it into an application if needed.
## PROGRAM

### Name:Mohanram Gunasekar
### Register Number:212223240095
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # 1 input channel (grayscale), 2 convolution blocks + fully connected layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # (32,28,28)
        self.pool  = nn.MaxPool2d(2, 2)                          # halves size
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # (64,14,14)

        # After 2 conv + pool layers: feature map size = 64 x 7 x 7
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)  # 10 classes in Fashion-MNIST

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)      # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)                # CrossEntropyLoss will apply Softmax
        return x

```

```python
from torchsummary import summary

# Initialize model
model = CNNClassifier()

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

# Print model summary
print('Name: Mohanram Gunasekar')
print('Register Number: 212223240095')
summary(model, input_size=(1, 28, 28))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

```python
def train_model(model, train_loader, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name: Mohanram Gunasekar')
        print('Register Number: 212223240095')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

## OUTPUT
### Training Loss per Epoch
<img width="325" height="202" alt="Screenshot 2025-09-02 113826" src="https://github.com/user-attachments/assets/cadff3e4-a426-47d5-a5e0-d86c2ed017b2" />

### Confusion Matrix
<img width="773" height="738" alt="Screenshot 2025-09-02 113335" src="https://github.com/user-attachments/assets/f40bc9cb-4c3d-4901-b58d-e9dc47b97a0a" />

### Classification Report
<img width="579" height="400" alt="Screenshot 2025-09-02 113349" src="https://github.com/user-attachments/assets/1d998970-3808-45e3-8168-9c076e3a536b" />

### New Sample Data Prediction
<img width="550" height="612" alt="Screenshot 2025-09-02 113142" src="https://github.com/user-attachments/assets/63c895b3-56a3-47b8-8ca6-78c5900c5e71" />

## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
