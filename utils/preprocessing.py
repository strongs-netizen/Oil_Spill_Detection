from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def preprocess(image):
    return transform(image).unsqueeze(0)   # add batch dimension → [1, 3, 224, 224]
