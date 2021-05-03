import torch
from PIL import Image
from torchvision import transforms

def infer(image):
    # Loading the image
    model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet121', pretrained=True)

    # Switching to eval mode for inference
    model.eval()

    # Defining path of the image
    file_path = build_file_path(image)

    # Opening the image
    input_image = Image.open(file_path)

    # Initialising the prediction variable
    prediction = "DEMO"
    
    # Image pre-processing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # Creating a mini-batch as expected by the model

    # Moving the input and model to GPU for speed, if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # The output has unnormalized scores. To get probabilities, we run a softmax on it
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)

    # Opening imagenet_classes.txt, reading all lines and removing white spaces
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    
    # Assigning class with maximum probability
    top_prob, top_catid = torch.topk(probabilities, 1)

    # Assigning prediction as the class with maximum probability
    prediction = categories[top_catid[0]]
    return prediction

def build_file_path(image):
    dir = "images/"
    file_path = dir + image
    return file_path