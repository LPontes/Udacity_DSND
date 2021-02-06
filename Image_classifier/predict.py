# Import libraries
import argparse
import torch
from torch.autograd import Variable
from torch import nn, optim
from torchvision import datasets, transforms, models
import numpy as np
import json
from PIL import Image
    

#set parser with argparser
parser = argparse.ArgumentParser(description = "Train a Convolutional Neural Network")
parser.add_argument('image_path')
parser.add_argument('checkpoint' , action = "store")
parser.add_argument('--top_k', type = int, default = 1)
parser.add_argument('--category_names', default = 'cat_to_name.json')
parser.add_argument('--gpu', action = 'store_true', default = False)


args = parser.parse_args()

category_names = args.category_names
top_k = args.top_k
image_path = args.image_path
checkpoint_path = args.checkpoint
device = 'cpu'
if args.gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

#Load model
def load_model(checkpoint_path):
    '''load a pre-trained model'''
    checkpoint = torch.load(checkpoint_path)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    
    if checkpoint['arch'] == 'vgg19':
        in_features = 25088
    elif checkpoint['arch'] == 'alexnet':
        in_features = 9216
    elif checkpoint['arch'] == 'densenet121':
        in_features = 1024
  
    lr = checkpoint['learning_rate']
    model.class_to_idx = checkpoint['class_to_idx']
    hidden_units = checkpoint['hidden_units']
    in_features =  model.classifier[0].in_features
    
    classifier = nn.Sequential(
        nn.Linear(in_features,hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.5),        
        nn.Linear(hidden_units,102),
        nn.LogSoftmax(dim=1)
    )
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    model.classifier = classifier
    model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    return model      
    
#Process image
def process_image(image):
    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    # Resize the images where shortest side is 256 pixels, keeping aspect ratio. 
    if image.width > image.height: 
        factor = image.width/image.height
        image = image.resize(size=(int(round(factor*256,0)),256))
    else:
        factor = image.height/image.width
        image = image.resize(size=(256, int(round(factor*256,0))))
    # Crop out the center 224x224 portion of the image.
    image = image.crop(box=((image.width/2)-112, (image.height/2)-112, (image.width/2)+112, (image.height/2)+112))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
       
    image = np.array(image)/255
    image = (image - mean)/std
    image = image.transpose((2,0,1))
    return image
    
def predict(image_path, model, topk= top_k):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = Image.open(image_path)
    model.to(device)
    #load the image
        
    inputs = torch.from_numpy(process_image(img)).float().unsqueeze(0)
    #load pre trained model
    model.eval()

    logps = model.forward(inputs.to(device))
    ps = torch.exp(logps).data
    top_p, top_class = ps.topk(topk, dim=1)
    top_class = list(top_class[0])
    top_p = top_p.tolist()[0]
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])
    # transfer index to label
    label = []
    for i in range(topk):
        label.append(ind[top_class[i]])
    
    return top_p, label

model = load_model(checkpoint_path)
probs, labels = predict(image_path, model)

print(cat_to_name[str(labels[0])])
print(labels)
print(probs)