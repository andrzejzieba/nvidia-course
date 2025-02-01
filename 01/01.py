import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# trainset example

train_set = torchvision.datasets.MNIST("./data/", train=True, download=True)
valid_set = torchvision.datasets.MNIST("./data/", train=False, download=True)

print(train_set)
print(valid_set)

x_0, y_0 = train_set[0]
print(x_0)
print(type(x_0))
print(y_0)

# tensor example

trans = transforms.Compose([transforms.ToTensor()])
x_0_tensor = trans(x_0)
print(x_0_tensor.dtype)
print(x_0_tensor.min(), x_0_tensor.max())
# C x H x W = 1 color channel, 28 x 28 pixels
print(x_0_tensor.size())
print(x_0_tensor)
print(x_0_tensor.device)
x_0_gpu = x_0_tensor.to(device)

# tensor to image
image = F.to_pil_image(x_0_tensor)
plt.imshow(image, cmap='gray')

train_set.transform = trans
valid_set.transform = trans

batch_size = 32

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)

input_size = 1 * 28 * 28 # C x H x W
n_classes = 10 # 0-9 numbers possible

# creating the model
layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512), # Input
    nn.ReLU(),  # Activation for input
    nn.Linear(512, 512),  # Hidden
    nn.ReLU(), # Activation for hidden
    nn.Linear(512, n_classes)  # Output
]

print(layers)

# compiling the model
model = nn.Sequential(*layers).to(device)

print(model)
print(next(model.parameters()).device)

# training the model

loss_function = nn.CrossEntropyLoss() # loss function to grade its answers
optimizer = Adam(model.parameters()) # tells the model how to learn from the loss

# In order to accurately calculate accuracy, we should compare the number of correct classifications compared to the total number of predictions made
train_N = len(train_loader.dataset)
valid_N = len(valid_loader.dataset)

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

# training the model

def train():
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

def validate():
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

# the training loop
epochs = 5

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()
    validate()

prediction = model(x_0_gpu)

# find the index of the highest value in the prediction tensor (ten elements)
print(prediction.argmax(dim=1, keepdim=True))
print(y_0)

# save the model
torch.save(model.state_dict(), 'model.pth')

# load the model
model = nn.Sequential(*layers).to(device)
model.load_state_dict(torch.load('model.pth'))

prediction = model(x_0_gpu)
print(prediction.argmax(dim=1, keepdim=True))
print(y_0)

