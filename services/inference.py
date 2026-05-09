import torch
import torch.nn.functional as F
from model.model import build_model

def load_model():
    model = build_model()
    model.load_state_dict(
        torch.load("model/neuralnet.pth", map_location="cpu")
    )
    model.eval()
    return model

def predict(model, tensor, classes):
    with torch.no_grad():
        output = model(tensor)
        probs  = F.softmax(output, dim=1)
        top_prob, top_class = torch.max(probs, 1)

    return classes[top_class.item()], top_prob.item()
