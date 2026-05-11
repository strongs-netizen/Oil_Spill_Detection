import os

import timm
import torch
import torch.nn.functional as F

MODEL_PATH = "oil_spill_model_weights.pth"
MODEL_NAME = "resnet50"
NUM_CLASSES = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path: str = MODEL_PATH):
    """Load the trained ResNet model weights for inference."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at '{model_path}'. "
            "Please place the trained file in the repository root."
        )

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict(model, tensor, classes):
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)
        top_prob, top_class = torch.max(probs, 1)

    return classes[top_class.item()], top_prob.item()
