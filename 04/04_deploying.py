import PIL
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.io as tv_io
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

from utils import MyConvBlock

model = torch.load('model.pth', map_location=device, weights_only=False)
print(model)

print(next(model.parameters()).device)

# input must match the shape of the data that the model was trained on
# the images in our dataset were 28x28 pixels and grayscale

image = tv_io.read_image('data/asl_images/b.png', tv_io.ImageReadMode.GRAY)
print(image.shape)

# Convert the image to float with ToDtype
# We will set scale to True in order to convert from [0, 255] to [0, 1]
# Resize the image to be 28 x 28 pixels
# Convert the images to Grayscale
# This step doesn't do anything since our models are already grayscale, but we've added it here to show an alternative way to get grayscale images.

IMG_WIDTH = 28
IMG_HEIGHT = 28

preprocess_trans = transforms.Compose([
    transforms.ToDtype(torch.float32, scale=True), # Converts [0, 255] to [0, 1]
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.Grayscale()  # From Color to Gray
])

processed_image = preprocess_trans(image)
print(processed_image.shape)

# model still expects a batch of images
batched_image = processed_image.unsqueeze(0)
print(batched_image.shape)

batched_image_gpu = batched_image.to(device)
print(batched_image_gpu.device)

# pass to model!
output = model(batched_image_gpu)
print(output)

# finding which element of the array represents the highest probability
prediction = output.argmax(dim=1).item()
print(prediction)

# Alphabet does not contain j or z because they require movement
alphabet = "abcdefghiklmnopqrstuvwxy"
print(alphabet[prediction])

def predict_letter(file_path):
    # Load and grayscale image
    image = tv_io.read_image(file_path, tv_io.ImageReadMode.GRAY)
    # Transform image
    image = preprocess_trans(image)
    # Save transformed image to _transformed file
    transformed_path = file_path.replace('.', '_transformed.')
    image_to_save = (image * 255).byte()
    tv_io.write_png(image_to_save, transformed_path)
    # Batch image
    image = image.unsqueeze(0)
    # Send image to correct device
    image = image.to(device)
    # Make prediction
    output = model(image)
    # Find max index
    prediction = output.argmax(dim=1).item()
    # Convert prediction to letter
    predicted_letter = alphabet[prediction]
    # Return prediction
    return predicted_letter

print(predict_letter("data/asl_images/b.png"))
print(predict_letter("data/asl_images/a.png"))
print(predict_letter("data/asl_images/k-test.png"))
print(predict_letter("data/asl_images/s-test.png"))