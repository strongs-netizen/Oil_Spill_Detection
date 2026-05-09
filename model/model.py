import torch.nn as nn

def build_model():
    model = nn.Sequential()

    # Conv Block 1
    model.append(nn.Conv2d(3, 16, kernel_size=3, padding=1))
    model.append(nn.ReLU())
    model.append(nn.MaxPool2d(2, 2))

    # Conv Block 2
    model.append(nn.Conv2d(16, 32, kernel_size=3, padding=1))
    model.append(nn.ReLU())
    model.append(nn.MaxPool2d(2, 2))

    # Conv Block 3
    model.append(nn.Conv2d(32, 64, kernel_size=3, padding=1))
    model.append(nn.ReLU())
    model.append(nn.MaxPool2d(2))

    # Classifier
    model.append(nn.Flatten())
    model.append(nn.Dropout())

    model.append(nn.Linear(50176, 500))
    model.append(nn.ReLU())
    model.append(nn.Dropout())

    model.append(nn.Linear(500, 2))   # 2 output classes: Oil Spill / No Oil Spill

    return model
