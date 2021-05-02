import torch
from PIL import Image
from torchvision import transforms

def infer(image):
    model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet121', pretrained=True)
    file_path = build_file_path(image)
    input_image = Image.open(file_path)
    trial = torch.tensor([[1., -1.], [1., -1.]])
    # print(trial)
    prediction = "DEMO"
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top_prob, top_catid = torch.topk(probabilities, 5)
    for i in range(top_prob.size(0)):
        print(categories[top_catid[i]], top_prob[i].item())
    prediction = categories[top_catid[0]]
    return prediction

def build_file_path(image):
    dir = "images/"
    file_path = dir + image
    return file_path