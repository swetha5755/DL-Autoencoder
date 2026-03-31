# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
This code implements a Denoising Autoencoder using PyTorch to clean noisy images from the MNIST dataset. It uses a convolutional neural network architecture, where the encoder compresses the input image into a lower-dimensional representation, and the decoder reconstructs the original image from this compressed form. To train the model to remove noise, Gaussian noise is added to the clean images, and the network learns to recover the original from the noisy version. The training process uses Mean Squared Error (MSE) as the loss function to measure the reconstruction error and the Adam optimizer to update the model weights. The autoencoder is trained over multiple epochs using mini-batches of data for efficiency. After training, the model's performance is visually evaluated by displaying the original, noisy, and denoised images side by side.

## DESIGN STEPS
### STEP 1:
Problem Understanding and Dataset Selection

### STEP 2: 
Preprocessing the Dataset

### STEP 3:
Design the Convolutional Autoencoder Architecture

### STEP 4:
Compile and Train the Model

### STEP 5: 
Evaluate the Model

### STEP 6: 
Visualization and Analysis


## PROGRAM

### Name: Swetha S

### Register Number:212224040344

```python
# Autoencoder Definition
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
      super(DenoisingAutoencoder, self).__init__()
      self.encoder = nn.Sequential(
          nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
          nn.ReLU()
      )
      self.decoder = nn.Sequential(
          nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
          nn.Sigmoid()
      )

    def forward(self, x):
      x = self.encoder(x)
      x = self.decoder(x)
      return x

# Initialize model

model = DenoisingAutoencoder().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training function

train(model, train_loader, criterion, optimizer, epochs=5)

# Visualization function

visualize_denoising(model, test_loader)
```

### OUTPUT

### Model Summary
<img width="511" height="403" alt="image" src="https://github.com/user-attachments/assets/cd914a09-2d7f-46dd-b704-fc8e40c42593" />


### Training loss
<img width="328" height="193" alt="image" src="https://github.com/user-attachments/assets/7360fb07-6b4d-40bc-a50b-562b1f90c1dc" />

## Original vs Noisy Vs Reconstructed Image

<img width="1734" height="326" alt="image" src="https://github.com/user-attachments/assets/e4ea6c21-0dd5-49d8-bc28-2360a7d08fb4" />
<img width="984" height="326" alt="image" src="https://github.com/user-attachments/assets/022c0c0d-c69c-46b0-8852-5c54c0d930b6" />

## RESULT
This program has been executed successfully.
