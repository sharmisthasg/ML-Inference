import torch
from PIL import Image
from torchvision import transforms

def infer(image):
    model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet121', pretrained=True)
    file_path = build_file_path(image)
    input_image = Image.open(file_path)
    trial = torch.tensor([[1., -1.], [1., -1.]])
    print(trial)
    prediction = "DEMO"
    '''
    TODO: Insert Inference Here
    '''
    return prediction

def build_file_path(image):
    dir = "images/"
    file_path = dir + image
    return file_path